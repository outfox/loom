from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loom.entry import Entry, StringEntry, FileEntry


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
        """Remove all entries from this section, releasing their IDs."""
        for entry in self.entries:
            entry.release()
        self.entries.clear()

    @property
    def has_multimodal(self) -> bool:
        """Check whether this section contains any multimodal entries."""
        return any(entry.content_blocks() is not None for entry in self.entries)

    def compile(self, seen: set[str] | None = None, exclude_roles: set[str] | None = None) -> str:
        """
        Compile all entries, optionally deduplicating against seen identities.

        Args:
            seen: Set of entry identities already compiled. Duplicates are skipped.
                  If None, no deduplication is performed.
            exclude_roles: Set of role names to exclude from compilation.
                  If None, all entries are included.

        Returns:
            Compiled string with prefix/postfix if section has content.
        """
        parts: list[str] = []
        for entry in self.entries:
            # Skip entries with excluded roles
            if exclude_roles and entry.role in exclude_roles:
                continue

            # Only perform deduplication if seen is provided
            if seen is not None:
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

    def compile_blocks(
        self,
        seen: set[str] | None = None,
        exclude_roles: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compile entries into a list of content blocks (text and/or image).

        Text entries are gathered into text blocks.  Multimodal entries
        (those whose ``content_blocks()`` returns a list) are emitted as
        image blocks interleaved with the surrounding text.

        Args:
            seen: Set of entry identities already compiled.  Duplicates
                are skipped.  If None, no deduplication is performed.
            exclude_roles: Set of role names to exclude from compilation.
                  If None, all entries are included.

        Returns:
            A list of Anthropic-style content block dicts.  Returns an
            empty list if the section has no (non-duplicate) content.
        """
        blocks: list[dict[str, Any]] = []
        text_parts: list[str] = []
        has_entries = False

        def flush_text() -> None:
            """Flush accumulated text parts into a single text block."""
            if not text_parts:
                return
            blocks.append({"type": "text", "text": "\n\n".join(text_parts)})
            text_parts.clear()

        for entry in self.entries:
            # Skip entries with excluded roles
            if exclude_roles and entry.role in exclude_roles:
                continue

            if seen is not None:
                ident = entry.identity()
                if ident in seen:
                    continue
                seen.add(ident)

            has_entries = True
            multimodal = entry.content_blocks()
            if multimodal is not None:
                # Flush any accumulated text before the image
                flush_text()
                blocks.extend(multimodal)
            else:
                compiled = entry.compile()
                if entry.name:
                    text_parts.append(f"# {entry.name}\n{compiled}")
                else:
                    text_parts.append(compiled)

        if not has_entries:
            return []

        flush_text()

        # Wrap with prefix/postfix
        if self.prefix and blocks:
            if blocks[0].get("type") == "text":
                blocks[0]["text"] = self.prefix + "\n" + blocks[0]["text"]
            else:
                blocks.insert(0, {"type": "text", "text": self.prefix})
        if self.postfix and blocks:
            if blocks[-1].get("type") == "text":
                blocks[-1]["text"] = blocks[-1]["text"] + "\n" + self.postfix
            else:
                blocks.append({"type": "text", "text": self.postfix})

        return blocks
