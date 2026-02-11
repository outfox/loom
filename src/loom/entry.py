"""
Entry - A piece of context that can be compiled into a string.

Entries wrap content sources (strings, files, etc.) and provide
metadata for deduplication and formatting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loom.ids import generate_id


class Entry(ABC):
    """Base class for context entries."""

    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: Human-readable identifier for this entry.
                  Used for deduplication and section headers.
        """
        self.name = name
        self.id = generate_id()
        self.created_at = datetime.now(timezone.utc)

    @abstractmethod
    def compile(self) -> str:
        """Compile this entry to a string."""
        ...

    @abstractmethod
    def identity(self) -> str:
        """
        Return an identity string for deduplication.
        Entries with the same identity are considered duplicates.
        """
        ...

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entry):
            return NotImplemented
        return self.identity() == other.identity()

    def __hash__(self) -> int:
        return hash(self.identity())


class StringEntry(Entry):
    """An entry backed by a plain string."""

    def __init__(self, content: str, name: Optional[str] = None):
        super().__init__(name)
        self._content = content

    def compile(self) -> str:
        return self._content

    def identity(self) -> str:
        # For strings, identity is the content hash
        return f"string:{hash(self._content)}"

    def __repr__(self) -> str:
        preview = self._content[:50] + "..." if len(self._content) > 50 else self._content
        return f"StringEntry({preview!r})"


class FileEntry(Entry):
    """An entry backed by a file on disk."""

    def __init__(self, path: str | Path, name: Optional[str] = None):
        self._path = Path(path)
        super().__init__(name or self._path.name)

    @property
    def path(self) -> Path:
        return self._path

    def compile(self) -> str:
        return self._path.read_text(encoding="utf-8")

    def identity(self) -> str:
        # For files, identity is the resolved path
        return f"file:{self._path.resolve()}"

    def __repr__(self) -> str:
        return f"FileEntry({self._path})"
