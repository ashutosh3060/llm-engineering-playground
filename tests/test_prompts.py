from __future__ import annotations

import pytest

from playground.prompts import FewShot, PromptTemplate


def test_variables_are_detected_in_system_and_user() -> None:
    tpl = PromptTemplate(name="t", system="Domain: $domain", user="Classify: ${text}")
    assert tpl.variables() == {"domain", "text"}


def test_missing_variable_raises_with_the_name() -> None:
    tpl = PromptTemplate(name="t", user="Hello $name, about $topic")
    with pytest.raises(KeyError, match="topic"):
        tpl.render(name="Ada")


def test_version_is_whitespace_sensitive() -> None:
    """A trailing space changes the bytes sent, so it is a different experiment."""
    tpl = PromptTemplate(name="t", user="Answer: $q")
    assert tpl.render(q="x").version != tpl.render(q="x ").version


def test_version_is_stable_across_renders() -> None:
    tpl = PromptTemplate(name="t", user="Answer: $q")
    assert tpl.render(q="same").version == tpl.render(q="same").version


def test_system_change_changes_version() -> None:
    a = PromptTemplate(name="t", system="Be terse.", user="hi").render()
    b = PromptTemplate(name="t", system="Be verbose.", user="hi").render()
    assert a.version != b.version


def test_few_shots_become_alternating_messages() -> None:
    tpl = PromptTemplate(
        name="t",
        user="Classify: $text",
        few_shots=[FewShot(user="great!", assistant="positive")],
    )
    msgs = tpl.render(text="awful").messages()
    assert [m.role for m in msgs] == ["user", "assistant", "user"]
    assert msgs[-1].content == "Classify: awful"


def test_few_shots_change_the_version() -> None:
    """Few-shot examples are part of the prompt, so they must affect its identity."""
    plain = PromptTemplate(name="t", user="x").render()
    shot = PromptTemplate(name="t", user="x", few_shots=[FewShot(user="a", assistant="b")]).render()
    assert plain.version != shot.version


def test_json_braces_survive_rendering() -> None:
    """Template uses $-substitution precisely so JSON schemas need no escaping."""
    tpl = PromptTemplate(name="t", user='Return {"name": string} for $text')
    rendered = tpl.render(text="Ada")
    assert '{"name": string}' in rendered.user


def test_to_request_carries_prompt_identity_into_metadata() -> None:
    version = PromptTemplate(name="cls", user="Classify $text").render(text="hi")
    request = version.to_request("mock-small", max_tokens=32, effort="low")
    assert request.metadata["prompt_name"] == "cls"
    assert request.metadata["prompt_version"] == version.version
    assert request.max_tokens == 32
