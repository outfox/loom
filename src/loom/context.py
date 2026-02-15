"""
Context - The main weaving loom for LLM context.

A Context manages multiple sections of entries and can include
other contexts as visitors, interleaving their content during compilation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from loom.compactor import Compactor
from loom.entry import Entry, FileEntry, StringEntry
from loom.ids import create_context_id
from loom.section import Section


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
        self.id = create_context_id()
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
        if visitor is self:
            raise ValueError("Cannot include a Context as its own visitor")
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

    def remember(self, entry: Entry | str | Path, *, summarize: bool = False) -> Entry:
        """
        Promote an entry to CONVO for long-term retention.

        Use this when something from the current step (or elsewhere) is important
        enough to remember beyond the current turn. This is how volatile work
        becomes persistent knowledge.

        Args:
            entry: The entry to remember. Strings and Paths are auto-wrapped.
            summarize: If True, prefix with "Learned: " (future: could use LLM).

        Returns:
            The entry that was added to CONVO.

        Example:
            # After an important tool call
            result = step.entries[-1]
            context.remember(result)

            # Or with a summary
            context.remember(f"The API returns {format}", summarize=True)
        """
        if isinstance(entry, str):
            # Don't add "Learned:" prefix here if summarize=True,
            # because the summarize block below will add it
            entry = StringEntry(entry)
        elif isinstance(entry, Path):
            entry = FileEntry(entry)

        # If summarizing, compile any Entry type into a summarized StringEntry
        if summarize and isinstance(entry, Entry):
            compiled = entry.compile()
            # Avoid double "Learned:" prefix
            if compiled.startswith("Learned: "):
                content = compiled
            else:
                content = f"Learned: {compiled}"
            entry = StringEntry(content, name=getattr(entry, "name", None))

        # If the entry exists in STEP, move it (remove without releasing ID)
        # to avoid sharing the same object between STEP and CONVO.
        # This prevents compile(clear_volatile=True) from releasing the ID
        # of an entry that's still in CONVO.
        if entry in self.step.entries:
            self.step.entries.remove(entry)

        self.convo.add(entry)
        return entry

    def _all_sections(self) -> list[Section]:
        """Return all sections in compilation order."""
        return [
            self.foundation,
            self.focus,
            self.topic,
            self.convo,
            self.step,
            self.attention,
        ]

    def entries(self, section: str | None = None) -> Iterator[Entry]:
        """
        Iterate over all entries in this context.

        Args:
            section: If provided, only yield entries from that section.
                     Use section names like "convo", "foundation", etc.

        Yields:
            Entry objects in section order.

        Example:
            # List all convo entries
            for entry in context.entries("convo"):
                print(f"{entry.id} | {entry.compile()[:50]}")
        """
        sections = self._all_sections()

        if section is not None:
            section_lower = section.lower()
            sections = [s for s in sections if s.name.lower() == section_lower]

        for sec in sections:
            yield from sec.entries

    def get(self, entry_id: str) -> Entry | None:
        """
        Find an entry by its ID.

        Args:
            entry_id: The short ID (e.g., "kvm", "axr")

        Returns:
            The Entry if found, None otherwise.

        Example:
            entry = context.get("kvm")
            if entry:
                context.redact(entry)
        """
        for entry in self.entries():
            if entry.id == entry_id:
                return entry
        return None

    def redact(
        self,
        entry: Entry | str,
        *,
        tombstone: str | None = None,
    ) -> bool:
        """
        Remove an entry from the context.

        Searches all sections for the entry and removes it. Optionally
        replaces it with a tombstone marker.

        Args:
            entry: The entry to remove, or its ID string.
            tombstone: If provided, replace the entry with this string instead
                       of removing it completely. Useful for preserving context
                       about why something was removed.

        Returns:
            True if the entry was found and removed, False otherwise.

        Example:
            # Complete removal
            context.redact(bad_entry)

            # By ID
            context.redact("kvm")

            # Leave a marker
            context.redact(pii_entry, tombstone="[REDACTED: contained PII]")
        """
        # Resolve ID to Entry if needed
        if isinstance(entry, str):
            resolved = self.get(entry)
            if resolved is None:
                return False
            entry = resolved

        for section in self._all_sections():
            for i, e in enumerate(section.entries):
                if e is entry:
                    if tombstone is not None:
                        # Replace with tombstone - release old entry's ID
                        entry.release()
                        section.entries[i] = StringEntry(tombstone, name=entry.name)
                    else:
                        # Complete removal - release ID back to pool
                        del section.entries[i]
                        entry.release()
                    return True

        return False

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

    def render(self, *, clear_volatile: bool = True) -> str:
        """
        Render the context to a string for LLM consumption.

        This is the primary public API for getting the compiled context.
        It's a convenience wrapper around compile() with sensible defaults.

        Args:
            clear_volatile: If True (default), clear the step section after rendering.

        Returns:
            The fully rendered context string, ready for use as a system prompt.

        Example:
            ctx = Context("main")
            ctx.foundation.add(StringEntry("You are a helpful assistant."))
            ctx.topic.add(FileEntry("project/README.md"))

            system_prompt = ctx.render()
            # Use system_prompt in your LLM API call
        """
        return self.compile(clear_volatile=clear_volatile)

    def to_messages(
        self,
        *,
        cache_breakpoints: list[str] | None = None,
        clear_volatile: bool = True,
    ) -> list[dict]:
        """
        Render the context as a list of messages for chat APIs.

        Returns a list suitable for OpenAI/Anthropic-style chat completions.
        Supports Anthropic's prompt caching via cache_breakpoints.

        Args:
            cache_breakpoints: List of section names after which to set cache
                breakpoints. Valid names: "foundation", "focus", "topic", "convo",
                "step", "attention". Max 4 breakpoints recommended (Anthropic limit).
                Example: ["foundation", "topic"] caches foundation and everything
                up to and including topic.
            clear_volatile: If True (default), clear the step section after rendering.

        Returns:
            List of message dicts. Without cache_breakpoints, returns simple format:
            [{"role": "system", "content": "..."}]

            With cache_breakpoints, returns Anthropic's multi-block format:
            [{"role": "system", "content": [
                {"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "..."},
                ...
            ]}]

        Example:
            ctx = Context("main")
            ctx.foundation.add(StringEntry("You are Blue, a helpful AI."))
            ctx.topic.add(StringEntry("Current task: help with Python"))

            # Simple format (OpenAI-compatible)
            messages = ctx.to_messages()

            # With caching (Anthropic-optimized)
            messages = ctx.to_messages(cache_breakpoints=["foundation", "topic"])
        """
        if cache_breakpoints is None:
            # Simple format - just compile everything
            system_content = self.compile(clear_volatile=clear_volatile)
            return [{"role": "system", "content": system_content}]

        # Multi-block format with cache breakpoints
        cache_set = {name.lower() for name in cache_breakpoints}
        seen: set[str] = set()
        content_blocks: list[dict] = []

        # Helper to add a content block
        def add_block(text: str, section_name: str) -> None:
            if not text.strip():
                return
            block: dict = {"type": "text", "text": text}
            if section_name.lower() in cache_set:
                block["cache_control"] = {"type": "ephemeral"}
            content_blocks.append(block)

        # FOUNDATION: self first, then visitors
        foundation_parts = []
        if content := self.foundation.compile(seen):
            foundation_parts.append(content)
        for visitor in self._visitors:
            if content := visitor.foundation.compile(seen):
                foundation_parts.append(f"># {visitor.name} (visitor)\n{content}")
        if foundation_parts:
            add_block("\n\n".join(foundation_parts), "foundation")

        # FOCUS: visitors first, then self
        focus_parts = []
        for visitor in self._visitors:
            if content := visitor.focus.compile(seen):
                focus_parts.append(f"># Focus from {visitor.name}\n{content}")
        if content := self.focus.compile(seen):
            focus_parts.append(content)
        if focus_parts:
            add_block("\n\n".join(focus_parts), "focus")

        # TOPIC: self first, then visitors
        topic_parts = []
        if content := self.topic.compile(seen):
            topic_parts.append(content)
        for visitor in self._visitors:
            if content := visitor.topic.compile(seen):
                topic_parts.append(f"># Topic from {visitor.name}\n{content}")
        if topic_parts:
            add_block("\n\n".join(topic_parts), "topic")

        # CONVO: visitors then self
        convo_parts = []
        for visitor in self._visitors:
            if content := visitor.convo.compile(seen):
                convo_parts.append(f"># Conversation from {visitor.name}\n{content}")
        if content := self.convo.compile(seen):
            convo_parts.append(content)
        if convo_parts:
            add_block("\n\n".join(convo_parts), "convo")

        # STEP: self only
        if content := self.step.compile():  # No dedup for step
            add_block(content, "step")

        # ATTENTION: visitors then self
        attention_parts = []
        for visitor in self._visitors:
            if content := visitor.attention.compile(seen):
                attention_parts.append(f"># Attention from {visitor.name}\n{content}")
        if content := self.attention.compile(seen):
            attention_parts.append(content)
        if attention_parts:
            add_block("\n\n".join(attention_parts), "attention")

        if clear_volatile:
            self.step.clear()

        return [{"role": "system", "content": content_blocks}]

    def __repr__(self) -> str:
        return f"Context({self.id!r}, {self.name!r}, visitors={len(self._visitors)})"
