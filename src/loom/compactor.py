"""
Compactor - Compress context using an LLM.

Compactors take a context and produce a condensed version,
preserving essential information while reducing token count.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.context import Context


class Compactor(ABC):
    """Base class for context compactors."""

    @abstractmethod
    def compact(self, context: "Context") -> str:
        """
        Compact the given context into a condensed string.

        Note: This should NOT compact visitor contexts - those are
        handled separately to preserve their structure.

        Args:
            context: The context to compact.

        Returns:
            A condensed string representation.
        """
        ...


class StubCompactor(Compactor):
    """
    A stub compactor that doesn't actually compact.

    *stubst dich aufs näschen* 🐾

    This is a placeholder until we implement real LLM-based compaction.
    """

    def compact(self, context: "Context") -> str:
        # For now, just return the compiled context unchanged
        # Real implementation would call an LLM to summarize
        return context.compile(clear_volatile=False)
