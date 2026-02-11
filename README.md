# 🧵 loom

*Weave context for LLM agents*

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

## Features

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
