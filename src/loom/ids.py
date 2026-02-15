"""
ID Generator for Entries.

Generates short, human-typeable IDs that appear in random order.
Uses a pre-shuffled pool for guaranteed uniqueness within a process.
"""

from __future__ import annotations

import random
from itertools import product

# Readable alphabet: no 0/O, 1/l/I confusion
# noinspection SpellCheckingInspection
ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # 31 chars (no 1)


class IDGenerator:
    """
    Generates unique, human-readable IDs.
    
    IDs are short and appear in scrambled order.
    The pool is pre-generated and shuffled at instantiation.
    
    Example:
        >>> gen = IDGenerator(length=3)
        >>> gen.next()
        'kvm'
        >>> gen.next()
        'axr'
    """

    def __init__(self, length: int, seed: int | None = None):
        """
        Args:
            length: Number of characters per ID (3 = 29,791 IDs, 4 = 923,521 IDs)
            seed: Random seed for reproducible shuffling (mainly for tests)
        
        Raises:
            ValueError: If length is not a positive integer.
        """
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"length must be a positive integer, got {length!r}")
        
        self.length = length
        
        # Generate all possible combinations
        all_ids = ["".join(combo) for combo in product(ALPHABET, repeat=length)]
        
        # Shuffle with optional seed
        rng = random.Random(seed)
        rng.shuffle(all_ids)
        
        self._pool = all_ids

    def next(self) -> str:
        """Get the next unique ID."""
        if not self._pool:
            raise RuntimeError(
                f"ID pool exhausted ({len(ALPHABET) ** self.length} IDs used). "
                "Consider using a longer length for bigger sessions."
            )
        return self._pool.pop()

    def release(self, entry_id: str) -> None:
        """Return an ID to the pool for reuse."""
        self._pool.append(entry_id)

    @property
    def remaining(self) -> int:
        """Number of IDs still available."""
        return len(self._pool)


# Global generators for the process
_generator_context: IDGenerator | None = None
_generator_entry: IDGenerator | None = None


def create_context_id() -> str:
    """Generate a unique context ID (3 chars = 29,791 contexts)."""
    global _generator_context
    if _generator_context is None:
        _generator_context = IDGenerator(length=3)
    return _generator_context.next()


def create_entry_id() -> str:
    """Generate a unique entry ID (4 chars = 923,521 entries)."""
    global _generator_entry
    if _generator_entry is None:
        _generator_entry = IDGenerator(length=4)
    return _generator_entry.next()


def release_id(entry_id: str) -> None:
    """Return an ID to the global entry pool for reuse."""
    global _generator_entry
    if _generator_entry is not None:
        _generator_entry.release(entry_id)


def reset_generator(seed: int | None = None, length: int = 4) -> None:
    """Reset the global entry generator. Mainly useful for testing."""
    global _generator_entry
    _generator_entry = IDGenerator(length=length, seed=seed)


def reset_context_generator(seed: int | None = None, length: int = 3) -> None:
    """Reset the global context generator. Mainly useful for testing."""
    global _generator_context
    _generator_context = IDGenerator(length=length, seed=seed)
