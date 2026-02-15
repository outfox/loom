"""Tests for Context IDs."""

import pytest
from loom import Context, reset_context_generator, reset_generator


class TestContextID:
    """Test that contexts get unique IDs."""

    def setup_method(self):
        """Reset generators for reproducible tests (length=2 for speed)."""
        reset_context_generator(seed=42, length=2)
        reset_generator(seed=42, length=2)

    def test_context_has_id(self):
        """Context should have an ID on creation."""
        ctx = Context("test")
        assert ctx.id is not None
        assert isinstance(ctx.id, str)
        assert len(ctx.id) == 2

    def test_context_ids_are_unique(self):
        """Each context should get a unique ID."""
        ctx1 = Context("first")
        ctx2 = Context("second")
        ctx3 = Context("third")

        ids = {ctx1.id, ctx2.id, ctx3.id}
        assert len(ids) == 3, "Context IDs should be unique"

    def test_context_id_in_repr(self):
        """Context repr should include the ID."""
        ctx = Context("mycontext")
        repr_str = repr(ctx)
        assert ctx.id in repr_str
        assert "mycontext" in repr_str

    def test_context_id_is_readable(self):
        """Context IDs should use the readable alphabet."""
        # Create several contexts and check their IDs
        contexts = [Context(f"ctx{i}") for i in range(10)]
        
        # Check that IDs only contain allowed characters
        allowed = set("abcdefghjkmnpqrstuvwxyz23456789")
        for ctx in contexts:
            assert all(c in allowed for c in ctx.id), f"Invalid char in {ctx.id}"
