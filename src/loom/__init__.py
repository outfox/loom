"""
loom - Weave context for LLM agents

A context management library that helps you compose, interleave, and compile
context from multiple sources into coherent prompts.
"""

from loom.compactor import Compactor
from loom.context import Context
from loom.entry import Entry, FileEntry, StringEntry
from loom.ids import IDGenerator, create_entry_id, create_context_id, reset_generator, reset_context_generator

__version__ = "0.1.0"
__all__ = [
    "Context",
    "Entry",
    "FileEntry",
    "StringEntry",
    "Compactor",
    "IDGenerator",
    "create_entry_id",
    "create_context_id",
    "reset_generator",
    "reset_context_generator",
]
