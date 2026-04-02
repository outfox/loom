"""Tests for ImageEntry and multimodal context support."""

import pytest

from loom import Context, ImageEntry, StringEntry, reset_generator, reset_context_generator


# Fake base64 data (doesn't need to be a real image for unit tests)
FAKE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
FAKE_MIME = "image/png"


@pytest.fixture(autouse=True)
def fast_generators():
    """Use short IDs for fast tests."""
    reset_generator(seed=42, length=2)
    reset_context_generator(seed=42, length=2)


class TestImageEntry:
    """Tests for ImageEntry basics."""

    def test_create_image_entry(self):
        entry = ImageEntry(FAKE_B64, FAKE_MIME, name="test image")
        assert entry.name == "test image"
        assert entry.data == FAKE_B64
        assert entry.media_type == FAKE_MIME

    def test_compile_returns_placeholder(self):
        entry = ImageEntry(FAKE_B64, FAKE_MIME, name="my photo")
        result = entry.compile()
        assert result == "[Image: my photo]"

    def test_compile_unnamed(self):
        entry = ImageEntry(FAKE_B64, FAKE_MIME)
        result = entry.compile()
        assert result == "[Image: unnamed]"

    def test_content_blocks_returns_image_block(self):
        entry = ImageEntry(FAKE_B64, FAKE_MIME, name="wolf")
        blocks = entry.content_blocks()

        assert blocks is not None
        assert len(blocks) == 2

        # Image block
        assert blocks[0]["type"] == "image"
        assert blocks[0]["source"]["type"] == "base64"
        assert blocks[0]["source"]["media_type"] == FAKE_MIME
        assert blocks[0]["source"]["data"] == FAKE_B64

        # Caption block
        assert blocks[1]["type"] == "text"
        assert "wolf" in blocks[1]["text"]

    def test_identity_is_stable(self):
        entry1 = ImageEntry(FAKE_B64, FAKE_MIME, name="a")
        entry2 = ImageEntry(FAKE_B64, FAKE_MIME, name="b")
        # Same data → same identity
        assert entry1.identity() == entry2.identity()

    def test_identity_differs_for_different_data(self):
        entry1 = ImageEntry("AAAA", FAKE_MIME)
        entry2 = ImageEntry("BBBB", FAKE_MIME)
        assert entry1.identity() != entry2.identity()

    def test_has_id_and_created_at(self):
        entry = ImageEntry(FAKE_B64, FAKE_MIME)
        assert entry.id is not None
        assert entry.created_at is not None

    def test_release_clears_id(self):
        entry = ImageEntry(FAKE_B64, FAKE_MIME)
        assert entry.id is not None
        entry.release()
        assert entry.id is None

    def test_repr(self):
        entry = ImageEntry(FAKE_B64, FAKE_MIME, name="wolf")
        r = repr(entry)
        assert "ImageEntry" in r
        assert "wolf" in r
        assert "image/png" in r

    def test_string_entry_content_blocks_is_none(self):
        """StringEntry.content_blocks() returns None (text-only)."""
        entry = StringEntry("Hello")
        assert entry.content_blocks() is None


class TestSectionMultimodal:
    """Tests for Section.compile_blocks() with multimodal entries."""

    def test_text_only_section(self):
        ctx = Context("test")
        ctx.foundation.add(StringEntry("Hello"))
        blocks = ctx.foundation.compile_blocks()

        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Hello" in blocks[0]["text"]

    def test_image_only_section(self):
        ctx = Context("test")
        ctx.topic.add(ImageEntry(FAKE_B64, FAKE_MIME, name="photo"))
        blocks = ctx.topic.compile_blocks()

        # Should have image block + caption
        image_blocks = [b for b in blocks if b["type"] == "image"]
        assert len(image_blocks) == 1
        assert image_blocks[0]["source"]["data"] == FAKE_B64

    def test_mixed_text_and_image(self):
        ctx = Context("test")
        ctx.topic.add(StringEntry("Before the image"))
        ctx.topic.add(ImageEntry(FAKE_B64, FAKE_MIME, name="photo"))
        ctx.topic.add(StringEntry("After the image"))

        blocks = ctx.topic.compile_blocks()

        # Should be: text, image, caption, text
        types = [b["type"] for b in blocks]
        assert "text" in types
        assert "image" in types

        # Text before image
        first_text = next(b for b in blocks if b["type"] == "text")
        assert "Before the image" in first_text["text"]

    def test_empty_section_returns_empty(self):
        ctx = Context("test")
        blocks = ctx.topic.compile_blocks()
        assert blocks == []

    def test_has_multimodal(self):
        ctx = Context("test")
        assert ctx.topic.has_multimodal() is False

        ctx.topic.add(ImageEntry(FAKE_B64, FAKE_MIME))
        assert ctx.topic.has_multimodal() is True

    def test_deduplication_in_compile_blocks(self):
        ctx = Context("test")
        ctx.topic.add(ImageEntry(FAKE_B64, FAKE_MIME, name="photo1"))
        ctx.topic.add(ImageEntry(FAKE_B64, FAKE_MIME, name="photo2"))  # Same data

        seen: set[str] = set()
        blocks = ctx.topic.compile_blocks(seen)

        # Only one image should appear (deduplicated by identity)
        image_blocks = [b for b in blocks if b["type"] == "image"]
        assert len(image_blocks) == 1


class TestContextMultimodalMessages:
    """Tests for Context.to_messages() with multimodal entries."""

    def test_text_only_returns_string_content(self):
        """Without images, to_messages() returns simple string format."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("You are helpful."))

        messages = ctx.to_messages()

        assert len(messages) == 1
        assert isinstance(messages[0]["content"], list)
        assert all(b["type"] == "text" for b in messages[0]["content"])

    def test_with_image_returns_block_format(self):
        """With images, to_messages() returns block format."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("You are helpful."))
        ctx.topic.add(ImageEntry(FAKE_B64, FAKE_MIME, name="reference"))

        messages = ctx.to_messages()

        assert len(messages) == 1
        assert isinstance(messages[0]["content"], list)

        # Should contain both text and image blocks
        types = {b["type"] for b in messages[0]["content"]}
        assert "text" in types
        assert "image" in types

    def test_cached_with_image(self):
        """Cache breakpoints work with multimodal content."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("Foundation text."))
        ctx.topic.add(ImageEntry(FAKE_B64, FAKE_MIME, name="diagram"))
        ctx.topic.add(StringEntry("Topic text."))

        messages = ctx.to_messages(cache_breakpoints=["foundation", "topic"])

        content = messages[0]["content"]
        assert isinstance(content, list)

        # Foundation should have cache_control
        # Find a block with cache_control
        cached_blocks = [b for b in content if "cache_control" in b]
        assert len(cached_blocks) >= 1

    def test_image_in_convo_persists(self):
        """Images remembered to convo persist across compiles."""
        ctx = Context("test")
        img = ImageEntry(FAKE_B64, FAKE_MIME, name="kept image")
        ctx.convo.add(img)

        # First compile
        messages1 = ctx.to_messages()
        assert isinstance(messages1[0]["content"], list)

        # Second compile — image should still be there
        messages2 = ctx.to_messages()
        assert isinstance(messages2[0]["content"], list)
        image_blocks = [b for b in messages2[0]["content"] if b["type"] == "image"]
        assert len(image_blocks) == 1

    def test_remember_image_entry(self):
        """context.remember() works with ImageEntry."""
        ctx = Context("test")
        img = ImageEntry(FAKE_B64, FAKE_MIME, name="important visual")
        ctx.step.add(img)

        # Remember it before step is cleared
        ctx.remember(img)

        # Compile clears step
        ctx.compile(clear_volatile=True)

        # But convo still has it
        assert len(ctx.convo.entries) == 1
        assert isinstance(ctx.convo.entries[0], ImageEntry)

    def test_redact_image_entry(self):
        """context.redact() works with ImageEntry."""
        ctx = Context("test")
        img = ImageEntry(FAKE_B64, FAKE_MIME, name="sensitive")
        ctx.convo.add(img)

        result = ctx.redact(img)
        assert result is True
        assert len(ctx.convo.entries) == 0

    def test_redact_image_with_tombstone(self):
        """Redacting an image with tombstone replaces it with text."""
        ctx = Context("test")
        img = ImageEntry(FAKE_B64, FAKE_MIME, name="classified")
        ctx.convo.add(img)

        ctx.redact(img, tombstone="[REDACTED: classified image]")

        assert len(ctx.convo.entries) == 1
        assert isinstance(ctx.convo.entries[0], StringEntry)
        assert ctx.convo.entries[0].compile() == "[REDACTED: classified image]"

    def test_compile_text_fallback_for_images(self):
        """compile() (text-only) uses placeholder for images."""
        ctx = Context("test")
        ctx.topic.add(ImageEntry(FAKE_B64, FAKE_MIME, name="photo"))

        result = ctx.compile()
        assert "[Image: photo]" in result

    def test_render_text_fallback_for_images(self):
        """render() (text-only) uses placeholder for images."""
        ctx = Context("test")
        ctx.topic.add(ImageEntry(FAKE_B64, FAKE_MIME, name="diagram"))

        result = ctx.render()
        assert "[Image: diagram]" in result


class TestContextMultimodalVisitors:
    """Tests for multimodal content with visitor contexts."""

    def test_visitor_image_included(self):
        """Visitor images are included in to_messages()."""
        main = Context("main")
        visitor = Context("visual")
        visitor.topic.add(ImageEntry(FAKE_B64, FAKE_MIME, name="visitor img"))

        main.include(visitor)
        main.foundation.add(StringEntry("Base"))

        messages = main.to_messages()
        content = messages[0]["content"]
        assert isinstance(content, list)

        image_blocks = [b for b in content if b["type"] == "image"]
        assert len(image_blocks) == 1

    def test_visitor_image_with_cache_breakpoints(self):
        """Visitor images work with cache breakpoints."""
        main = Context("main")
        visitor = Context("visual")
        visitor.foundation.add(ImageEntry(FAKE_B64, FAKE_MIME, name="logo"))

        main.include(visitor)
        main.foundation.add(StringEntry("Identity"))

        messages = main.to_messages(cache_breakpoints=["foundation"])
        content = messages[0]["content"]
        assert isinstance(content, list)

        # Should have cache_control on last foundation block
        cached = [b for b in content if "cache_control" in b]
        assert len(cached) >= 1
