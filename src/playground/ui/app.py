"""Streamlit UI.

Four tabs, matching the four questions this project exists to answer:

  Comparison     which model should serve this prompt?
  Prompt Lab     which prompt/parameter combination is best?
  Cost Analyzer  what does a prompt cost before I run it at scale?
  Runs           what have I already measured?

The audience is engineers evaluating models, not end users — which is why this is
Streamlit and not a production frontend. That work is Month 5.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from ai_core.cost import cost_breakdown, estimate_cost
from ai_core.models import REGISTRY
from ai_core.providers.base import ProviderUnavailable
from ai_core.types import CompletionRequest, Message, Usage

# Allow `streamlit run src/playground/ui/app.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playground.benchmark import BenchmarkSuite, run_benchmark  # noqa: E402
from playground.prompts import PromptTemplate  # noqa: E402
from playground.runtime import build_gateway, get_store, provider_report  # noqa: E402
from playground.stats import percentile  # noqa: E402
from playground.sweep import SweepTooLarge, expand, grid_size  # noqa: E402

st.set_page_config(page_title="LLM Engineering Playground", page_icon="🧪", layout="wide")


@st.cache_resource
def _gateway():  # type: ignore[no-untyped-def]
    return build_gateway()


@st.cache_resource
def _store():  # type: ignore[no-untyped-def]
    return get_store()


gw = _gateway()
store = _store()
available = gw.available_models()

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("🧪 Playground")
    st.caption("Model comparison, prompt lab, and cost analysis.")

    st.subheader("Providers")
    for row in provider_report():
        icon = {"ok": "🟢", "error": "🔴", "no-key": "⚪"}[str(row["state"])]
        st.write(f"{icon} **{row['name']}** — {row['detail']}")

    if any(m.startswith("mock-") for m in available):
        st.warning(
            "The offline **mock** provider is active. Its output is synthetic and "
            "deterministic — useful for exercising the app without a key, but never "
            "quote its numbers as results.",
            icon="⚠️",
        )

    if not available:
        st.error("No models available. Set `ANTHROPIC_API_KEY` in `.env`.")
        st.stop()

    st.divider()
    st.metric("Recorded spend", f"${store.total_spend():.4f}")

tab_compare, tab_lab, tab_cost, tab_runs = st.tabs(
    ["Comparison", "Prompt Lab", "Cost Analyzer", "Runs"]
)

# --------------------------------------------------------------- comparison

with tab_compare:
    st.header("Model comparison")
    st.caption(
        "One prompt, every model, dispatched concurrently. Latency is a percentile "
        "over repeats — a single sample is noise."
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        system = st.text_area("System prompt", value="", height=80, key="cmp_sys")
        prompt = st.text_area(
            "User prompt",
            value="Explain prompt caching to a senior engineer in two sentences.",
            height=140,
            key="cmp_user",
        )
    with col_b:
        models = st.multiselect("Models", available, default=available[:3], key="cmp_models")
        repeats = st.slider("Repeats per model", 1, 10, 3, key="cmp_repeats")
        max_tokens = st.number_input("Max tokens", 16, 8192, 512, key="cmp_maxtok")
        effort = st.selectbox(
            "Effort", ["(default)", "low", "medium", "high", "xhigh", "max"], key="cmp_effort"
        )

    st.caption(f"This will issue **{len(models) * repeats}** calls.")

    if st.button("Run comparison", type="primary", disabled=not models):
        base = CompletionRequest(
            model=models[0],
            messages=[Message(role="user", content=prompt)],
            system=system or None,
            max_tokens=int(max_tokens),
            effort=None if effort == "(default)" else effort,  # type: ignore[arg-type]
        )
        run_id = store.start_run("cmp-ui-" + base.prompt_hash(), kind="compare", models=models)
        rows, samples = [], {}
        bar = st.progress(0.0, text="running...")
        collected: dict[str, list] = {}

        for rep in range(repeats):
            try:
                for result in gw.compare(base, models=models):
                    collected.setdefault(result.model, []).append(result)
                    store.record(run_id, result, repeat=rep)
            except ProviderUnavailable as exc:
                st.error(str(exc))
                break
            bar.progress((rep + 1) / repeats, text=f"repeat {rep + 1}/{repeats}")
        store.finish_run(run_id)
        bar.empty()

        for model, results in collected.items():
            ok = [r for r in results if r.ok]
            lat = sorted(r.latency_ms for r in ok)
            rows.append(
                {
                    "model": model,
                    "ok": len(ok),
                    "errors": len(results) - len(ok),
                    "$/call": round(sum(r.cost_usd for r in ok) / len(ok), 6) if ok else 0,
                    "p50 ms": round(percentile(lat, 50), 1),
                    "p95 ms": round(percentile(lat, 95), 1),
                    "out tokens": sum(r.usage.output_tokens for r in ok),
                }
            )
            if ok:
                samples[model] = ok[0].text

        df = pd.DataFrame(rows).sort_values("$/call").reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        if len(df) > 1:
            cheap, dear = df.iloc[0], df.iloc[-1]
            if cheap["$/call"] > 0:
                st.info(
                    f"**{cheap['model']}** costs "
                    f"{dear['$/call'] / cheap['$/call']:.0f}x less per call than "
                    f"**{dear['model']}**. Whether that trade is acceptable is what "
                    f"the benchmark suites answer."
                )
        st.bar_chart(df.set_index("model")[["p50 ms"]])

        st.subheader("Responses")
        for model, text in samples.items():
            with st.expander(model):
                st.write(text)

# --------------------------------------------------------------- prompt lab

with tab_lab:
    st.header("Prompt laboratory")
    st.caption(
        "Prompts are versioned by content hash of the rendered text — a trailing "
        "space is a different version, because it is a different experiment."
    )

    lab_system = st.text_area(
        "System prompt", value="You are a concise technical writer.", height=80, key="lab_sys"
    )
    lab_user = st.text_area(
        "User prompt (use $variable for placeholders)",
        value="Summarise the following in one sentence:\n\n$text",
        height=120,
        key="lab_user",
    )
    template = PromptTemplate(name="lab", system=lab_system or None, user=lab_user)
    variables = sorted(template.variables())

    values = {}
    if variables:
        st.write("**Variables**")
        for var in variables:
            values[var] = st.text_input(f"${var}", value="", key=f"lab_var_{var}")

    try:
        version = template.render(**values)
        st.code(f"prompt version: {version.version}", language=None)
        with st.expander("Rendered prompt"):
            if version.system:
                st.text(f"[system]\n{version.system}\n")
            st.text(f"[user]\n{version.user}")
    except KeyError as exc:
        st.warning(str(exc))
        version = None

    st.divider()
    st.subheader("Parameter sweep")
    sweep_models = st.multiselect("Models", available, default=available[:2], key="lab_models")
    sweep_efforts = st.multiselect(
        "Effort levels",
        ["low", "medium", "high", "xhigh", "max"],
        default=["low", "high"],
        key="lab_effort",
    )
    sweep_repeats = st.slider("Repeats per cell", 1, 5, 2, key="lab_repeats")

    axes = {"model": sweep_models, "effort": sweep_efforts}
    if sweep_models and sweep_efforts:
        st.caption(
            f"Grid: {grid_size(axes)} cells x {sweep_repeats} repeats = "
            f"**{grid_size(axes) * sweep_repeats}** calls."
        )

    if st.button("Run sweep", disabled=not (version and sweep_models and sweep_efforts)):
        try:
            cells = expand(axes, repeats=sweep_repeats)
        except SweepTooLarge as exc:
            st.error(str(exc))
            cells = []

        if cells and version:
            run_id = store.start_run(f"sweep-{version.version}", kind="sweep", label="prompt-lab")
            rows = []
            bar = st.progress(0.0, text="running...")
            for i, cell in enumerate(cells, start=1):
                request = version.to_request(cell["model"], max_tokens=512, effort=cell["effort"])
                result = gw.complete(request)
                store.record(run_id, result, prompt_label=f"lab:{version.version}")
                rows.append(
                    {
                        "model": cell["model"],
                        "effort": cell["effort"],
                        "latency_ms": round(result.latency_ms, 1),
                        "cost_usd": round(result.cost_usd, 6),
                        "out_tokens": result.usage.output_tokens,
                        "ok": result.ok,
                        "text": result.text[:160],
                    }
                )
                bar.progress(i / len(cells), text=f"{i}/{len(cells)}")
            store.finish_run(run_id)
            bar.empty()

            df = pd.DataFrame(rows)
            agg = (
                df[df.ok]
                .groupby(["model", "effort"], as_index=False)
                .agg(
                    cost_usd=("cost_usd", "mean"),
                    latency_ms=("latency_ms", "median"),
                    out_tokens=("out_tokens", "mean"),
                )
                .sort_values("cost_usd")
            )
            st.dataframe(agg, use_container_width=True, hide_index=True)
            with st.expander("Raw cells"):
                st.dataframe(df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------- cost analyzer

with tab_cost:
    st.header("Token & cost analyzer")
    st.caption(
        "Counts come from the provider's own tokenizer. `tiktoken` is OpenAI's and "
        "undercounts Claude by 15-20% on prose, more on code — estimates built on it "
        "are wrong in a way that only surfaces on the bill."
    )

    col_l, col_r = st.columns(2)
    with col_l:
        cost_model = st.selectbox("Model", available, key="cost_model")
        cost_system = st.text_area("System prompt", value="", height=100, key="cost_sys")
        cost_user = st.text_area("User prompt + context", value="", height=200, key="cost_user")
        expected_output = st.number_input("Expected output tokens", 0, 128_000, 500, key="cost_out")
        scale = st.number_input("Calls per month", 1, 10_000_000, 10_000, key="cost_scale")

    with col_r:
        if cost_user.strip():
            request = CompletionRequest(
                model=cost_model,
                messages=[Message(role="user", content=cost_user)],
                system=cost_system or None,
            )
            try:
                input_tokens = gw.count_tokens(request)
            except (ProviderUnavailable, Exception) as exc:  # noqa: BLE001
                st.error(f"Token count failed: {exc}")
                input_tokens = 0

            if input_tokens:
                usage = Usage(input_tokens=input_tokens, output_tokens=int(expected_output))
                per_call = estimate_cost(cost_model, usage)
                st.metric("Input tokens", f"{input_tokens:,}")
                st.metric("Cost per call", f"${per_call:.6f}")
                st.metric(f"Cost per {scale:,} calls", f"${per_call * scale:,.2f}")

                st.write("**Breakdown per call**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"component": k, "usd": round(v, 8)}
                            for k, v in cost_breakdown(cost_model, usage).items()
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

                st.write(f"**Same usage across every tier** (at {scale:,} calls/month)")
                today = date.today()
                tier_rows = []
                for mid, spec in sorted(REGISTRY.items()):
                    if spec.provider == "mock":
                        continue
                    monthly = estimate_cost(mid, usage, today) * scale
                    tier_rows.append(
                        {"model": mid, "tier": spec.tier, "monthly_usd": round(monthly, 2)}
                    )
                tiers = pd.DataFrame(tier_rows).sort_values("monthly_usd")
                st.dataframe(tiers, hide_index=True, use_container_width=True)
                if len(tiers) > 1 and tiers.iloc[0]["monthly_usd"] > 0:
                    saving = tiers.iloc[-1]["monthly_usd"] - tiers.iloc[0]["monthly_usd"]
                    st.success(
                        f"Serving this workload on **{tiers.iloc[0]['model']}** instead of "
                        f"**{tiers.iloc[-1]['model']}** saves **${saving:,.2f}/month**. "
                        f"Run a benchmark suite to find out whether quality holds."
                    )
        else:
            st.info("Paste a prompt on the left to see its token count and cost.")

# ---------------------------------------------------------------------- runs

with tab_runs:
    st.header("Recorded runs")
    st.caption("Every call this playground has made, with tokens, latency, and cost.")

    runs = store.runs(limit=100)
    if not runs:
        st.info("No runs yet. Use the Comparison or Prompt Lab tab, or `playground bench`.")
    else:
        st.dataframe(
            pd.DataFrame(runs)[["id", "kind", "label", "started_at", "n_results", "total_cost"]],
            use_container_width=True,
            hide_index=True,
        )
        picked = st.selectbox("Inspect run", [r["id"] for r in runs])
        if picked:
            detail = pd.DataFrame(store.results(picked))
            if not detail.empty:
                st.dataframe(
                    detail[
                        [
                            "case_id",
                            "model",
                            "repeat",
                            "input_tokens",
                            "output_tokens",
                            "latency_ms",
                            "cost_usd",
                            "score",
                            "error",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()
        st.subheader("Spend by model")
        spend = store.spend_by_model()
        if spend:
            st.dataframe(pd.DataFrame(spend), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Run a benchmark suite")
    suite_dir = Path(__file__).resolve().parents[3] / "datasets"
    suites = sorted(suite_dir.glob("*.yaml")) if suite_dir.exists() else []
    if suites:
        chosen = st.selectbox("Suite", suites, format_func=lambda p: p.stem)
        bench_models = st.multiselect(
            "Models", available, default=available[:2], key="bench_models"
        )
        bench_repeats = st.slider("Repeats", 1, 10, 3, key="bench_repeats")
        suite = BenchmarkSuite.from_yaml(chosen)
        st.caption(
            f"{len(suite.cases)} cases x {len(bench_models)} models x {bench_repeats} "
            f"repeats = **{len(suite.cases) * len(bench_models) * bench_repeats}** calls."
        )
        if st.button("Run benchmark", disabled=not bench_models):
            bar = st.progress(0.0, text="running...")
            _, summaries, _ = run_benchmark(
                suite,
                bench_models,
                gateway=gw,
                store=store,
                repeats=bench_repeats,
                progress=lambda d, t: bar.progress(d / t, text=f"{d}/{t}"),
            )
            bar.empty()
            st.dataframe(
                pd.DataFrame([s.as_row() for s in summaries]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No suites found in `datasets/`.")
