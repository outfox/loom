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

    @staticmethod
    def _visitor_label(section_name: str, visitor_name: str) -> str:
        """Return the visitor prefix label for a given section."""
        if section_name == "foundation":
            return f"># {visitor_name} (visitor)"
        labels = {
            "focus": "Focus",
            "topic": "Topic",
            "convo": "Conversation",
            "attention": "Attention",
        }
        word = labels.get(section_name, section_name)
        return f"># {word} from {visitor_name}"

    def _compile_sections(
        self,
        seen: set[str],
        exclude_roles: set[str] | None = None,
        clear_volatile: bool = True,
    ) -> str:
        """
        Shared compilation logic for interleaving visitor contexts.

        Args:
            seen: Set of already-seen entry identities for deduplication.
            exclude_roles: If provided, entries with these roles are skipped.
            clear_volatile: If True, clear the step section after compilation.

        Returns:
            The fully compiled context string.
        """
        parts: list[str] = []

        label = self._visitor_label

        # FOUNDATION: self first, then visitors
        if content := self.foundation.compile(seen, exclude_roles=exclude_roles):
            parts.append(content)
        for visitor in self._visitors:
            if content := visitor.foundation.compile(seen, exclude_roles=exclude_roles):
                parts.append(f"{label('foundation', visitor.name)}\n{content}")

        # FOCUS: visitors first, then self (inverted!)
        for visitor in self._visitors:
            if content := visitor.focus.compile(seen, exclude_roles=exclude_roles):
                parts.append(f"{label('focus', visitor.name)}\n{content}")
        if content := self.focus.compile(seen, exclude_roles=exclude_roles):
            parts.append(content)

        # TOPIC: self first, then visitors
        if content := self.topic.compile(seen, exclude_roles=exclude_roles):
            parts.append(content)
        for visitor in self._visitors:
            if content := visitor.topic.compile(seen, exclude_roles=exclude_roles):
                parts.append(f"{label('topic', visitor.name)}\n{content}")

        # CONVO: visitors (potentially compacted) then self (always full)
        for visitor in self._visitors:
            if content := visitor.convo.compile(seen, exclude_roles=exclude_roles):
                parts.append(f"{label('convo', visitor.name)}\n{content}")
        if content := self.convo.compile(seen, exclude_roles=exclude_roles):
            parts.append(content)

        # STEP: self only, always
        if content := self.step.compile(exclude_roles=exclude_roles):  # No dedup for step
            parts.append(content)

        # ATTENTION: visitors then self
        for visitor in self._visitors:
            if content := visitor.attention.compile(seen, exclude_roles=exclude_roles):
                parts.append(f"{label('attention', visitor.name)}\n{content}")
        if content := self.attention.compile(seen, exclude_roles=exclude_roles):
            parts.append(content)

        if clear_volatile:
            self.step.clear()

        return "\n\n".join(parts)

    def compile(self, clear_volatile: bool = True) -> str:
        """
        Compile the full context, interleaving visitor contexts.

        Args:
            clear_volatile: If True, clear the step section after compilation.

        Returns:
            The fully compiled context string.
        """
        return self._compile_sections(set(), clear_volatile=clear_volatile)

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
        cache_ttl: int | None = None,
        sections: list[str] | None = None,
    ) -> list[dict]:
        """
        Render the context as a list of messages for chat APIs.

        Returns a list suitable for OpenAI/Anthropic-style chat completions.
        Supports Anthropic's prompt caching via cache_breakpoints.

        When the context contains multimodal entries (e.g. ``ImageEntry``),
        their content blocks are emitted inline so that vision-capable
        models can see them directly in the system prompt.

        Entries with a non-system role (e.g. ``role="assistant"``) are
        emitted as separate messages after the system message but before
        any conversation history.

        Args:
            cache_breakpoints: List of section names after which to set cache
                breakpoints. Valid names: "foundation", "focus", "topic", "convo",
                "step", "attention". Max 4 breakpoints recommended (Anthropic limit).
                Example: ["foundation", "topic"] caches foundation and everything
                up to and including topic.
            clear_volatile: If True (default), clear the step section after rendering.
            cache_ttl: Optional TTL in seconds for Anthropic's prompt cache
                (``max_age_seconds``). When set, each ``cache_control`` block
                includes ``{"type": "ephemeral", "max_age_seconds": cache_ttl}``.
            sections: If provided, only include these sections (by name) in the
                output. Defaults to all sections. Useful for excluding volatile
                sections (step, attention) that the caller handles separately.

        Returns:
            List of message dicts. Without cache_breakpoints, returns simple format:
            [{"role": "system", "content": [<content blocks>]}]

            With cache_breakpoints, returns Anthropic's multi-block format:
            [{"role": "system", "content": [
                {"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "..."},
                ...
            ]}]

            Non-system entries are emitted as additional messages after the
            system message.

        Example:
            ctx = Context("main")
            ctx.foundation.add(StringEntry("You are Blue, a helpful AI."))
            ctx.topic.add(StringEntry("Current task: help with Python"))

            # Simple format (OpenAI-compatible)
            messages = ctx.to_messages()

            # With caching (Anthropic-optimized)
            messages = ctx.to_messages(cache_breakpoints=["foundation", "topic"])

            # Exclude volatile sections
            messages = ctx.to_messages(
                cache_breakpoints=["foundation", "topic"],
                sections=["foundation", "focus", "topic", "convo"],
            )
        """
        section_order = sections or ["foundation", "focus", "topic", "convo", "step", "attention"]

        if cache_breakpoints is None:
            return self._to_messages_simple(clear_volatile, section_order)

        return self._to_messages_cached(
            cache_breakpoints, clear_volatile, cache_ttl, section_order
        )

    def _collect_non_system_roles(self) -> set[str]:
        """Collect all non-system roles used by entries in this context and visitors."""
        roles: set[str] = set()
        for section in self._all_sections():
            for entry in section.entries:
                if entry.role != "system":
                    roles.add(entry.role)
        for visitor in self._visitors:
            for section in visitor._all_sections():
                for entry in section.entries:
                    if entry.role != "system":
                        roles.add(entry.role)
        return roles

    def _collect_non_system_entries(self) -> list[tuple[str, str]]:
        """Collect deduplicated entries with non-system roles. Returns (role, content) tuples."""
        results: list[tuple[str, str]] = []
        seen: set[str] = set()
        for section in self._all_sections():
            self._collect_non_system_from(section, results, seen)
        for visitor in self._visitors:
            for section in visitor._all_sections():
                self._collect_non_system_from(section, results, seen)
        return results

    @staticmethod
    def _collect_non_system_from(
        section: Section,
        results: list[tuple[str, str]],
        seen: set[str],
    ) -> None:
        """Collect non-system entries from a single section into results, deduplicating via seen."""
        for entry in section.entries:
            ident = entry.identity()
            if ident in seen:
                continue
            # Compile first — FileEntry.compile() may update entry.role from frontmatter
            content = entry.compile()
            if entry.role == "system":
                continue
            seen.add(ident)
            if entry.name:
                content = f"# {entry.name}\n{content}"
            results.append((entry.role, content))

    def _has_multimodal(self, exclude_roles: set[str] | None = None) -> bool:
        """Check whether any section (self or visitors) has multimodal content.

        Args:
            exclude_roles: If provided, ignore entries with these roles when checking.
        """
        for section in self._all_sections():
            if section.has_multimodal(exclude_roles):
                return True
        for visitor in self._visitors:
            for section in visitor._all_sections():
                if section.has_multimodal(exclude_roles):
                    return True
        return False

    def _compile_system_only(
        self, clear_volatile: bool, non_system_roles: set[str] | None = None,
    ) -> str:
        """Compile the context excluding non-system entries."""
        if non_system_roles is None:
            non_system_roles = self._collect_non_system_roles()
        if not non_system_roles:
            return self.compile(clear_volatile=clear_volatile)
        return self._compile_sections(
            set(), exclude_roles=non_system_roles, clear_volatile=clear_volatile
        )

    def _to_messages_simple(
        self, clear_volatile: bool, section_order: list[str],
    ) -> list[dict]:
        """Build messages without cache breakpoints."""
        non_system_entries = self._collect_non_system_entries()
        non_system_roles = self._collect_non_system_roles() if non_system_entries else None

        seen: set[str] = set()
        all_blocks: list[dict] = []

        for section_name in section_order:
            if section_name == "step":
                all_blocks.extend(self.step.compile_blocks(exclude_roles=non_system_roles))
            else:
                all_blocks.extend(
                    self._compile_section_group_blocks(section_name, seen, exclude_roles=non_system_roles)
                )

        if clear_volatile:
            self.step.clear()

        messages: list[dict] = [{"role": "system", "content": all_blocks}]

        # Add non-system entries grouped by role (block format)
        if non_system_entries:
            grouped: list[tuple[str, list[dict]]] = []
            for role, content in non_system_entries:
                block: dict = {"type": "text", "text": content}
                if grouped and grouped[-1][0] == role:
                    grouped[-1][1].append(block)
                else:
                    grouped.append((role, [block]))
            for role, blocks in grouped:
                messages.append({"role": role, "content": blocks})

        return messages

    def _to_messages_cached(
        self,
        cache_breakpoints: list[str],
        clear_volatile: bool,
        cache_ttl: int | None = None,
        section_order: list[str] | None = None,
    ) -> list[dict]:
        """Build messages with cache breakpoints (always block format)."""
        if section_order is None:
            section_order = ["foundation", "focus", "topic", "convo", "step", "attention"]
        cache_set = {name.lower() for name in cache_breakpoints}
        non_system_entries = self._collect_non_system_entries()
        non_system_roles = self._collect_non_system_roles() if non_system_entries else None
        seen: set[str] = set()
        content_blocks: list[dict] = []

        for section_name in section_order:
            if section_name == "step":
                # Step: self only, no dedup
                blocks = self.step.compile_blocks(exclude_roles=non_system_roles)
            else:
                blocks = self._compile_section_group_blocks(section_name, seen, exclude_roles=non_system_roles)

            if not blocks:
                continue

            # Apply cache_control to the last block of this section group
            if section_name in cache_set:
                cc: dict = {"type": "ephemeral"}
                if cache_ttl is not None:
                    cc["max_age_seconds"] = cache_ttl
                blocks[-1] = {**blocks[-1], "cache_control": cc}

            content_blocks.extend(blocks)

        if clear_volatile:
            self.step.clear()

        messages: list[dict] = [{"role": "system", "content": content_blocks}]

        # Add non-system entries grouped by role, each as a block-format message.
        # Cache breakpoint on the last block of the last message.
        if non_system_entries:
            cc_ns: dict = {"type": "ephemeral"}
            if cache_ttl is not None:
                cc_ns["max_age_seconds"] = cache_ttl

            # Group consecutive entries by role
            grouped: list[tuple[str, list[dict]]] = []
            for role, content in non_system_entries:
                block: dict = {"type": "text", "text": content}
                if grouped and grouped[-1][0] == role:
                    grouped[-1][1].append(block)
                else:
                    grouped.append((role, [block]))

            # Cache breakpoint on the very last block
            grouped[-1][1][-1]["cache_control"] = cc_ns

            for role, blocks in grouped:
                messages.append({"role": role, "content": blocks})

        return messages

    def _compile_section_group_blocks(
        self,
        section_name: str,
        seen: set[str],
        exclude_roles: set[str] | None = None,
    ) -> list[dict]:
        """
        Compile a section group (self + visitors) into content blocks.

        Each entry is emitted as its own content block, giving LLMs cleaner
        context boundaries between prompt sections.

        Follows the same ordering as compile():
        - foundation: self first, then visitors
        - focus: visitors first, then self
        - topic: self first, then visitors
        - convo: visitors first, then self
        - attention: visitors first, then self
        """
        self_section: Section = getattr(self, section_name)

        visitor_sections = [
            (v, getattr(v, section_name)) for v in self._visitors
        ]

        blocks: list[dict] = []

        if section_name in ("foundation", "topic"):
            # Self first, then visitors
            blocks.extend(self_section.compile_blocks(seen, exclude_roles=exclude_roles))
            for visitor, vsec in visitor_sections:
                vblocks = vsec.compile_blocks(seen, exclude_roles=exclude_roles)
                if vblocks:
                    vblocks.insert(0, {"type": "text", "text": self._visitor_label(section_name, visitor.name)})
                    blocks.extend(vblocks)
        else:
            # Visitors first, then self (focus, convo, attention)
            for visitor, vsec in visitor_sections:
                vblocks = vsec.compile_blocks(seen, exclude_roles=exclude_roles)
                if vblocks:
                    vblocks.insert(0, {"type": "text", "text": self._visitor_label(section_name, visitor.name)})
                    blocks.extend(vblocks)
            blocks.extend(self_section.compile_blocks(seen, exclude_roles=exclude_roles))

        return blocks

    def __repr__(self) -> str:
        return f"Context({self.id!r}, {self.name!r}, visitors={len(self._visitors)})"
