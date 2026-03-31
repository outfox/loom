"""
Entry - A piece of context that can be compiled into a string.

Entries wrap content sources (strings, files, images, etc.) and provide
metadata for deduplication and formatting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

from loom.ids import create_entry_id, release_id


class Entry(ABC):
    """Base class for context entries."""

    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: Human-readable identifier for this entry.
                  Used for deduplication and section headers.
        """
        self.name = name
        self.id = create_entry_id()
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

    def content_blocks(self) -> list[dict[str, Any]] | None:
        """
        Return multimodal content blocks for this entry, or None.

        Override in subclasses that carry non-text content (e.g. images).
        When this returns a list, Section and Context will emit content
        blocks instead of plain text for this entry.

        Returns:
            A list of Anthropic-style content block dicts, or None for
            text-only entries (the default).
        """
        return None

    def release(self) -> None:
        """Release this entry's ID back to the pool for reuse."""
        if self.id:
            release_id(self.id)
            self.id = None

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


class ImageEntry(Entry):
    """
    An entry backed by a base64-encoded image.

    When compiled to plain text (e.g. for render()), returns a placeholder
    like ``[Image: name]``.  When used in a multimodal context (e.g.
    to_messages()), the image data is emitted as an Anthropic-style
    content block via content_blocks().
    """

    def __init__(
        self,
        data: str,
        media_type: str,
        name: Optional[str] = None,
    ):
        """
        Args:
            data: Base64-encoded image data.
            media_type: MIME type (e.g. "image/jpeg", "image/png").
            name: Human-readable label for the image.
        """
        super().__init__(name)
        self._data = data
        self._media_type = media_type

    @property
    def data(self) -> str:
        """The base64-encoded image data."""
        return self._data

    @property
    def media_type(self) -> str:
        """The MIME type of the image."""
        return self._media_type

    def compile(self) -> str:
        """Return a text placeholder for plain-text contexts."""
        label = self.name or "unnamed"
        return f"[Image: {label}]"

    def content_blocks(self) -> list[dict[str, Any]]:
        """Return the image as an Anthropic-style content block."""
        blocks: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self._media_type,
                    "data": self._data,
                },
            },
        ]
        # Add a text caption so the model knows which image this is
        label = self.name or "unnamed"
        blocks.append({"type": "text", "text": f"[Kept image: {label}]"})
        return blocks

    def identity(self) -> str:
        """Identity based on a hash of the image data (first 256 chars)."""
        # Use a stable hash of the data prefix for deduplication
        digest = sha256(self._data[:256].encode("ascii")).hexdigest()[:16]
        return f"image:{digest}"

    def __repr__(self) -> str:
        label = self.name or "unnamed"
        return f"ImageEntry({label!r}, {self._media_type})"
