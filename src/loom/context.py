"""
Context - The main weaving loom for LLM context.

A Context manages multiple sections of entries and can include
other contexts as visitors, interleaving their content during compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loom.compactor import Compactor
from loom.entry import Entry, FileEntry, StringEntry


@dataclass
class Section:
    """A named section containing a list of entries."""

    name: str
    entries: list[Entry] = field(default_factory=list)
    prefix: str = ""
    postfix: str = ""

    def add(self, entry: Entry | str | Path) -> Entry:
        """Add an entry to this section. Strings and Paths are auto-wrapped."""
        if isinstance(entry, str):
            entry = StringEntry(entry)
        elif isinstance(entry, Path):
            entry = FileEntry(entry)
        self.entries.append(entry)
        return entry

    def clear(self) -> None:
        """Remove all entries from this section."""
        self.entries.clear()

    def compile(self, seen: set[str] | None = None) -> str:
        """
        Compile all entries, optionally deduplicating against seen identities.

        Args:
            seen: Set of entry identities already compiled. Duplicates are skipped.
                  If None, no deduplication is performed.

        Returns:
            Compiled string with prefix/postfix if section has content.
        """
        if seen is None:
            seen = set()

        parts: list[str] = []
        for entry in self.entries:
            identity = entry.identity()
            if identity in seen:
                continue
            seen.add(identity)

            compiled = entry.compile()
            if entry.name:
                parts.append(f"# {entry.name}\n{compiled}")
            else:
                parts.append(compiled)

        if not parts:
            return ""

        content = "\n\n".join(parts)
        result = ""
        if self.prefix:
            result += self.prefix + "\n"
        result += content
        if self.postfix:
            result += "\n" + self.postfix
        return result


class Context:
    """
    A context loom that weaves together multiple sections and visitor contexts.

    Sections are compiled in order:
    - foundation (self, then visitors)
    - focus (visitors first, then self)
    - topic (self, then visitors)
    - convo (visitors simplified, then self full)
    - step (self only, cleared after compile)
    - attention (visitors, then self)
    """

    def __init__(
        self,
        name: str,
        router: Optional[str] = None,
        cwd: Optional[str | Path] = None,
    ):
        """
        Args:
            name: Human-readable name for this context.
            router: LLM endpoint (e.g. litellm model string) this context prefers.
            cwd: Current working directory for file resolution.
        """
        self.name = name
        self.router = router
        self.cwd = Path(cwd) if cwd else Path.cwd()

        # Sections in compilation order
        self.foundation = Section("FOUNDATION", prefix="> FOUNDATION")
        self.focus = Section("FOCUS", prefix="------\n> FOCUS", postfix="------")
        self.topic = Section("TOPIC", prefix="------\n> TOPIC", postfix="------")
        self.convo = Section("CONVO", prefix="> CONVO - what has been said and done and learned?")
        self.step = Section("STEP")  # Volatile - cleared after compile
        self.attention = Section("ATTENTION", prefix="> ATTENTION")

        # Visitor contexts
        self._visitors: list[Context] = []

    @property
    def visitors(self) -> list[Context]:
        """Read-only view of visitor contexts."""
        return list(self._visitors)

    def include(self, visitor: Context) -> None:
        """Add a visitor context to be interleaved during compilation."""
        if visitor not in self._visitors:
            self._visitors.append(visitor)

    def exclude(self, visitor: Context) -> None:
        """Remove a visitor context."""
        if visitor in self._visitors:
            self._visitors.remove(visitor)

    def compile(self, clear_volatile: bool = True) -> str:
        """
        Compile the full context, interleaving visitor contexts.

        Args:
            clear_volatile: If True, clear the step section after compilation.

        Returns:
            The fully compiled context string.
        """
        seen: set[str] = set()
        parts: list[str] = []

        # FOUNDATION: self first, then visitors
        if content := self.foundation.compile(seen):
            parts.append(content)
        for visitor in self._visitors:
            if content := visitor.foundation.compile(seen):
                parts.append(f"># {visitor.name} (visitor)\n{content}")

        # FOCUS: visitors first, then self (inverted!)
        for visitor in self._visitors:
            if content := visitor.focus.compile(seen):
                parts.append(f"># Focus from {visitor.name}\n{content}")
        if content := self.focus.compile(seen):
            parts.append(content)

        # TOPIC: self first, then visitors
        if content := self.topic.compile(seen):
            parts.append(content)
        for visitor in self._visitors:
            if content := visitor.topic.compile(seen):
                parts.append(f"># Topic from {visitor.name}\n{content}")

        # CONVO: visitors (potentially compacted) then self (always full)
        for visitor in self._visitors:
            if content := visitor.convo.compile(seen):
                parts.append(f"># Conversation from {visitor.name}\n{content}")
        if content := self.convo.compile(seen):
            parts.append(content)

        # STEP: self only, always
        if content := self.step.compile():  # No dedup for step
            parts.append(content)

        # ATTENTION: visitors then self
        for visitor in self._visitors:
            if content := visitor.attention.compile(seen):
                parts.append(f"># Attention from {visitor.name}\n{content}")
        if content := self.attention.compile(seen):
            parts.append(content)

        if clear_volatile:
            self.step.clear()

        return "\n\n".join(parts)

    def compact(self, compactor: Compactor) -> str:
        """
        Compact this context using the given compactor.

        Note: Does not compact visitor contexts.

        Args:
            compactor: The compactor to use.

        Returns:
            Compacted context string.
        """
        return compactor.compact(self)

    def __repr__(self) -> str:
        return f"Context({self.name!r}, visitors={len(self._visitors)})"
