"""Tests for ID generation."""

import pytest
from loom import IDGenerator, reset_generator, create_entry_id
from loom.ids import release_id, ALPHABET


class TestIDGenerator:
    def test_requires_explicit_length(self):
        """IDGenerator requires explicit length - no magic defaults."""
        with pytest.raises(TypeError):
            IDGenerator(seed=42)  # Missing required 'length'

    def test_generates_unique_ids(self):
        gen = IDGenerator(length=3, seed=42)
        ids = [gen.next() for _ in range(100)]
        
        assert len(ids) == len(set(ids))  # All unique

    def test_ids_are_correct_length(self):
        gen = IDGenerator(length=3, seed=42)
        id3 = gen.next()
        assert len(id3) == 3

        gen4 = IDGenerator(length=2, seed=42)
        id4 = gen4.next()
        assert len(id4) == 2

    def test_ids_use_readable_alphabet(self):
        gen = IDGenerator(length=3, seed=42)
        
        # Generate many IDs and check characters
        for _ in range(1000):
            id = gen.next()
            for char in id:
                # No confusing characters
                assert char not in "01ilo"
                # Only valid chars
                assert char in ALPHABET

    def test_ids_appear_scrambled(self):
        gen = IDGenerator(length=3, seed=42)
        
        # First few IDs should not be alphabetically sorted
        ids = [gen.next() for _ in range(10)]
        assert ids != sorted(ids)

    def test_remaining_count(self):
        gen = IDGenerator(length=3, seed=42)
        initial = gen.remaining
        
        assert initial == len(ALPHABET) ** 3  # 31^3 = 29791
        
        gen.next()
        assert gen.remaining == initial - 1

    def test_exhaustion_raises(self):
        gen = IDGenerator(length=1, seed=42)  # Only 31 IDs
        
        # Use all IDs
        for _ in range(len(ALPHABET)):
            gen.next()
        
        with pytest.raises(RuntimeError, match="exhausted"):
            gen.next()

    def test_seed_makes_reproducible(self):
        gen1 = IDGenerator(length=3, seed=123)
        gen2 = IDGenerator(length=3, seed=123)
        
        ids1 = [gen1.next() for _ in range(10)]
        ids2 = [gen2.next() for _ in range(10)]
        
        assert ids1 == ids2

    def test_different_seeds_different_order(self):
        gen1 = IDGenerator(length=3, seed=1)
        gen2 = IDGenerator(length=3, seed=2)
        
        ids1 = [gen1.next() for _ in range(10)]
        ids2 = [gen2.next() for _ in range(10)]
        
        assert ids1 != ids2


class TestIDRelease:
    """Tests for ID recycling."""

    def test_release_returns_id_to_pool(self):
        gen = IDGenerator(length=3, seed=42)
        initial_remaining = gen.remaining

        id1 = gen.next()
        assert gen.remaining == initial_remaining - 1

        gen.release(id1)
        assert gen.remaining == initial_remaining

    def test_released_id_can_be_reused(self):
        gen = IDGenerator(length=3, seed=42)

        id1 = gen.next()
        gen.release(id1)

        # Released ID goes to end of pool, will be next
        id2 = gen.next()
        assert id2 == id1


class TestGlobalGenerator:
    def test_generate_id_works(self):
        reset_generator(seed=42, length=2)
        
        id1 = create_entry_id()
        id2 = create_entry_id()
        
        assert id1 != id2
        assert len(id1) == 2
        assert len(id2) == 2

    def test_reset_generator(self):
        reset_generator(seed=42, length=2)
        id1 = create_entry_id()
        
        reset_generator(seed=42, length=2)
        id2 = create_entry_id()
        
        assert id1 == id2  # Same seed = same first ID
