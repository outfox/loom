from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loom import Entry, StringEntry, FileEntry


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

    def compile(self, seen: set[str] | None = None) -> str:
        """
        Compile all entries, optionally deduplicating against seen identities.

        Args:
            seen: Set of entry identities already compiled. Duplicates are skipped.
                  If None, no deduplication is performed.

        Returns:
            Compiled string with prefix/postfix if section has content.
        """
        parts: list[str] = []
        for entry in self.entries:
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
