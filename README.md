![loom logo](logo-loom.png)

# LOOM

A Python library for composing, interleaving, and compiling context from multiple sources into coherent prompts.

## Installation

```bash
pip install context-loom
```

## Quick Start

```python
from loom import Context

# Create a context
ctx = Context("my-agent", router="anthropic/claude-sonnet-4-20250514")

# Add foundation (like a system prompt)
ctx.foundation.add("You are a helpful assistant.")
ctx.foundation.add("./SOUL.md")  # Files work too!

# Add focus (current task)
ctx.focus.add("Review PR #42 and suggest improvements.")

# Include visitor contexts (they get interleaved)
security_ctx = Context("security")
security_ctx.foundation.add("Always validate user input.")
ctx.include(security_ctx)

# Compile everything together
prompt = ctx.compile()
```

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

## Why This Order? Prefix Caching!

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

All major LLM providers (Anthropic, OpenAI, etc.) use **prefix caching** — they cache the beginning of your prompt and only recompute from where it changes.

By putting stable content first and volatile content last:
- **90%+ cache hit rate** on typical requests
- **Lower latency** (less to recompute)
- **Lower cost** (cached tokens are cheaper or free)

The `step` section is cleared after each `compile(clear_volatile=True)` — but everything before it stays identical, maximizing cache reuse.

## Features

- **Cache-Optimized**: Section order designed for maximum prefix cache hits.
- **Deduplication**: Same entry in multiple contexts? Only compiled once.
- **Visitors**: Include other contexts, interleaved intelligently.
- **Volatile sections**: `step` is cleared after each `compile()`.
- **File entries**: Point to files, they're read at compile time.
- **Compaction**: (Coming soon) Compress context with LLMs.

## Development

### Guidelines

- **Test Coverage:** We aim for excellent test coverage. Run `pytest` after every change.
- **String Literals:** Be careful with character escaping — context compilation can surface tricky edge cases. Tests help catch these.

### Running Tests

```bash
pytest
pytest --cov=loom  # with coverage
```

## License

AGPL-3.0-or-later
