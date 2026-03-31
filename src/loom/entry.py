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


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from content. Returns (metadata, body)."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("---", 3)
    if end == -1:
        return {}, content
    front = content[3:end].strip()
    body = content[end + 3:].lstrip("\n")
    meta = {}
    for line in front.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body


class Entry(ABC):
    """Base class for context entries."""

    def __init__(self, name: Optional[str] = None, role: str = "system"):
        """
        Args:
            name: Human-readable identifier for this entry.
                  Used for deduplication and section headers.
            role: Message role for this entry (e.g. "system", "assistant").
                  Defaults to "system".
        """
        self.name = name
        self.role = role
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

    def __init__(self, content: str, name: Optional[str] = None, role: str = "system"):
        super().__init__(name, role=role)
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
    """An entry backed by a file on disk.

    Re-reads the file on every ``compile()`` so edits are reflected
    immediately.  If the file is deleted or becomes unreadable after the
    entry is created, ``compile()`` returns a short notice instead of
    raising.
    """

    _SENTINEL = object()

    def __init__(self, path: str | Path, name: Optional[str] = None, role: object = _SENTINEL):
        self._path = Path(path)
        self._explicit_role = role
        self._frontmatter: dict[str, str] = {}
        self._body: str = ""

        # Initial read — sets role from frontmatter when no explicit role given
        try:
            self._read()
        except (FileNotFoundError, OSError):
            pass  # File will be read on first compile()

        # Determine role: explicit parameter > frontmatter > default "system"
        if role is not FileEntry._SENTINEL:
            resolved_role = role  # type: ignore[assignment]
        elif "role" in self._frontmatter:
            resolved_role = self._frontmatter["role"]
        else:
            resolved_role = "system"

        super().__init__(name or self._path.name, role=resolved_role)

    def _read(self) -> None:
        """Read file content and parse frontmatter."""
        raw = self._path.read_text(encoding="utf-8")
        self._frontmatter, self._body = _parse_frontmatter(raw)

    @property
    def path(self) -> Path:
        return self._path

    def compile(self) -> str:
        try:
            self._read()
        except FileNotFoundError:
            return f"[File removed: {self._path.name}]"
        except OSError as e:
            return f"[File unreadable: {self._path.name} ({e})]"

        # Re-resolve role from frontmatter on each read (explicit overrides still win)
        if self._explicit_role is FileEntry._SENTINEL:
            self.role = self._frontmatter.get("role", "system")

        return self._body

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
        role: str = "system",
    ):
        """
        Args:
            data: Base64-encoded image data.
            media_type: MIME type (e.g. "image/jpeg", "image/png").
            name: Human-readable label for the image.
            role: Message role for this entry. Defaults to "system".
        """
        super().__init__(name, role=role)
        self._data = data
        self._media_type = media_type
        self._identity = f"image:{sha256(data.encode('ascii')).hexdigest()}"

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
        """Identity based on a hash of the full image data."""
        return self._identity

    def __repr__(self) -> str:
        label = self.name or "unnamed"
        return f"ImageEntry({label!r}, {self._media_type})"
