![loom logo](logo-loom.png)

# LOOM

A Python library for composing, interleaving, and compiling context from multiple sources into coherent prompts.

## Installation

Loom is not yet published to PyPI. Install from git:

```bash
uv add git+https://github.com/outfox/loom
# or
pip install git+https://github.com/outfox/loom
```

A PyPI release under `context-loom` is planned.

## Quick Start

```python
from loom import Context, StringEntry, FileEntry

ctx = Context("my-agent")

# Stable identity & rules
ctx.foundation.add(StringEntry("You are a helpful assistant."))
ctx.foundation.add(FileEntry("./SOUL.md"))  # re-read on every compile

# Current task
ctx.focus.add(StringEntry("Review PR #42 and suggest improvements."))

# Pull in another context — its sections are interleaved with this one
security_ctx = Context("security")
security_ctx.foundation.add(StringEntry("Always validate user input."))
ctx.include(security_ctx)

# Render as Anthropic-style messages with explicit cache breakpoints
messages = ctx.to_messages(
    cache_breakpoints=["foundation", "topic"],
    cache_ttl=3600,  # optional; sets Anthropic's max_age_seconds
)

# Drop straight into your LLM call
# response = client.messages.create(model="claude-sonnet-4-6", system=messages[0]["content"], ...)
```

For plain-text output (e.g. dumping into a single system prompt), use `ctx.render()` instead.

## Sections

Context is organized into sections, compiled in this order:

| Section | Purpose | Visitor Order |
|---------|---------|---------------|
| `foundation` | Core identity, rules | self → visitors |
| `focus` | Current task/skill | visitors → self |
| `topic` | What we're working on | self → visitors |
| `convo` | Conversation history | visitors → self |
| `step` | Current command output | self only (volatile) |
| `attention` | Reinforcement/reminders | visitors → self |

## Why this order? Prefix caching

The section order isn't arbitrary — it's optimized for **LLM prefix caching**:

```text
┌─────────────────────────────────┐
│ FOUNDATION (stable)             │  ← Cached
│ FOCUS (relatively stable)       │  ← Cached
│ TOPIC (session-stable)          │  ← Cached
│ CONVO (grows, but append-only)  │  ← Partially cached
├─────────────────────────────────┤
│ STEP (volatile, changes often)  │  ← Never cached
│ ATTENTION (volatile)            │  ← Never cached
└─────────────────────────────────┘
```

All major LLM providers cache the beginning of your prompt and only recompute from where it changes. By putting stable content first and volatile content last, you get high cache hit rates, lower latency, and lower cost.

`to_messages(cache_breakpoints=[...])` makes this explicit: each named section gets a `cache_control: {"type": "ephemeral"}` marker on its last block, and `cache_ttl=N` sets `max_age_seconds=N`. The `step` section is cleared after each compile, so everything before it stays byte-identical between turns.

## Features

### Cache-aware message rendering

`to_messages()` produces Anthropic/OpenAI-style chat messages, with one content block per entry — cleaner section boundaries for the model than a single concatenated string.

```python
messages = ctx.to_messages(
    cache_breakpoints=["foundation", "topic"],  # up to 4
    cache_ttl=3600,                             # optional max_age_seconds
    sections=["foundation", "focus", "topic", "convo"],  # selective
)
```

`sections=` lets you exclude volatile sections (e.g. when the caller manages `step` separately).

### Visitors with deduplication

`ctx.include(other)` interleaves another context's sections into this one's compile order. The same entry referenced from multiple contexts is compiled exactly once.

```python
shared = Context("shared-rules")
shared.foundation.add(FileEntry("./RULES.md"))

agent_a.include(shared)
agent_b.include(shared)
# RULES.md is compiled once per agent; if both feed into a parent context,
# deduplication kicks in there too.
```

### File entries with frontmatter

`FileEntry` re-reads the file on every compile, so edits show up without rebuilding the context. A YAML frontmatter block can promote the file out of the system prompt:

```markdown
---
role: assistant
---
Here is what I remember from our last session...
```

In `to_messages()`, that file is emitted as its own `assistant` message after the system block, instead of being inlined into the system prompt.

### Multimodal

`ImageEntry` carries base64 image data and a media type. In `to_messages()` it's emitted inline as a vision content block (with a text caption so the model can refer back to it). In plain `render()` it falls back to `[Image: name]`.

```python
from loom import ImageEntry

ctx.topic.add(ImageEntry(data=b64, media_type="image/png", name="screenshot"))
```

### Lifecycle: `remember()`, `redact()`, entry IDs

Every entry gets a short, human-typeable id (e.g. `"kvm"`, `"axr"`) and a creation timestamp. IDs are recycled back into the pool when entries are removed, so long-lived processes don't exhaust them.

```python
# Promote a transient STEP entry into CONVO for long-term retention
result = ctx.step.add(StringEntry(tool_output))
ctx.remember(result)

# Remove an entry by id, optionally leaving a tombstone
ctx.redact("kvm", tombstone="[REDACTED: contained PII]")
```

### Volatile `step`

The `step` section is cleared after every `compile()` / `to_messages()` call. Pass `clear_volatile=False` to keep it across calls.

## Compaction (planned)

A `Compactor` abstract base class lives in `loom.compactor`. The shipped implementation (`StubCompactor`) is a passthrough — it returns the compiled context unchanged. An LLM-backed compactor that summarizes older `convo` entries is the next milestone.

## Development

The package layout is `src/loom/`; tests are in `tests/`.

- **Test coverage:** run `pytest` after every change.
- **String literals:** be careful with escaping — context compilation surfaces tricky edge cases. Tests help catch these.

```bash
pytest
pytest --cov=loom  # with coverage
```

## License

AGPL-3.0-or-later
