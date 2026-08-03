"""Prompt versioning.

A prompt is an experimental variable, so it is identified by the hash of its
*rendered* text — not by a filename. Two runs share a version only if the exact
bytes sent to the model were identical, including whitespace. That is what makes
a comparison attributable to the prompt rather than to an unnoticed edit.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from string import Template
from typing import Any

from ai_core.types import CompletionRequest, Message

__all__ = ["FewShot", "PromptTemplate", "PromptVersion", "render"]

_VAR = re.compile(r"\$\{?(\w+)\}?")


@dataclass(frozen=True)
class FewShot:
    user: str
    assistant: str


@dataclass
class PromptTemplate:
    """A named, parameterised prompt.

    Uses ``string.Template`` (``$var`` / ``${var}``) rather than str.format so that
    JSON braces in a prompt body do not need escaping — prompts containing JSON
    schemas are common and ``{}`` collisions are a constant annoyance otherwise.
    """

    name: str
    user: str
    system: str | None = None
    few_shots: list[FewShot] = field(default_factory=list)
    description: str = ""

    def variables(self) -> set[str]:
        text = f"{self.system or ''}\n{self.user}"
        return set(_VAR.findall(text))

    def render(self, **values: Any) -> PromptVersion:
        missing = self.variables() - set(values)
        if missing:
            raise KeyError(f"Prompt {self.name!r} needs variable(s): {', '.join(sorted(missing))}")
        system = Template(self.system).safe_substitute(values) if self.system else None
        user = Template(self.user).safe_substitute(values)
        return PromptVersion(
            name=self.name,
            system=system,
            user=user,
            few_shots=list(self.few_shots),
            values=dict(values),
        )


@dataclass(frozen=True)
class PromptVersion:
    """A fully-rendered prompt, identified by content hash."""

    name: str
    user: str
    system: str | None = None
    few_shots: list[FewShot] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)

    def messages(self) -> list[Message]:
        msgs: list[Message] = []
        for shot in self.few_shots:
            msgs.append(Message(role="user", content=shot.user))
            msgs.append(Message(role="assistant", content=shot.assistant))
        msgs.append(Message(role="user", content=self.user))
        return msgs

    @property
    def version(self) -> str:
        """Content hash over everything that reaches the model.

        Deliberately whitespace-sensitive: a trailing space changes the bytes sent,
        so it changes the version. Treating those as the same prompt would silently
        merge two different experiments.
        """
        parts = [self.system or ""]
        for shot in self.few_shots:
            parts.extend([shot.user, shot.assistant])
        parts.append(self.user)
        blob = "\x00".join(parts)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_request(
        self,
        model: str,
        *,
        max_tokens: int = 2048,
        effort: str | None = None,
        thinking: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionRequest:
        return CompletionRequest(
            model=model,
            messages=self.messages(),
            system=self.system,
            max_tokens=max_tokens,
            effort=effort,  # type: ignore[arg-type]
            thinking=thinking,
            metadata={"prompt_name": self.name, "prompt_version": self.version, **(metadata or {})},
        )


def render(template: PromptTemplate, **values: Any) -> PromptVersion:
    return template.render(**values)
