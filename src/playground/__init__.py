"""llm-engineering-playground — compare models on quality, latency, and cost.

    from playground.runtime import build_gateway, get_store
    from playground.benchmark import BenchmarkSuite, run_benchmark

    suite = BenchmarkSuite.from_yaml("datasets/sentiment-classification.yaml")
    run_id, summaries, _ = run_benchmark(suite, ["claude-haiku-4-5"], repeats=5)

CLI: `playground probe | models | bench | spend | serve | ui`
"""

__version__ = "0.1.0"
