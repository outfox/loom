"""
loom - Weave context for LLM agents

A context management library that helps you compose, interleave, and compile
context from multiple sources into coherent prompts.
"""

from loom.context import Context
from loom.entry import Entry, FileEntry, StringEntry
from loom.compactor import Compactor

__version__ = "0.1.0"
__all__ = ["Context", "Entry", "FileEntry", "StringEntry", "Compactor"]
