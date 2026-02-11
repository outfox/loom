"""
ID Generator for Entries.

Generates short, human-typeable IDs that appear in random order.
Uses a pre-shuffled pool for guaranteed uniqueness within a process.
"""

from __future__ import annotations

import random
from itertools import product

# Readable alphabet: no 0/O, 1/l/I confusion
ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # 30 chars


class IDGenerator:
    """
    Generates unique, human-readable IDs.
    
    IDs are short (default 3 chars) and appear in scrambled order.
    The pool is pre-generated and shuffled at instantiation.
    
    Example:
        >>> gen = IDGenerator()
        >>> gen.next()
        'kvm'
        >>> gen.next()
        'axr'
    """

    def __init__(self, length: int = 3, seed: int | None = None):
        """
        Args:
            length: Number of characters per ID (3 = 27k IDs, 4 = 810k IDs)
            seed: Random seed for reproducible shuffling (mainly for tests)
        """
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
                "Consider using length=4 for longer sessions."
            )
        return self._pool.pop()

    @property
    def remaining(self) -> int:
        """Number of IDs still available."""
        return len(self._pool)


# Global generator for the process
_generator: IDGenerator | None = None


def generate_id() -> str:
    """Generate a unique ID using the global generator."""
    global _generator
    if _generator is None:
        _generator = IDGenerator()
    return _generator.next()


def reset_generator(seed: int | None = None, length: int = 3) -> None:
    """Reset the global generator. Mainly useful for testing."""
    global _generator
    _generator = IDGenerator(length=length, seed=seed)
