"""Memory module for agent integration.

Provides memory capabilities for the SWE-Agent, enabling:
- Recall: Retrieve relevant experience before reasoning
- Retain: Store learnings from successful actions

Uses actual classes from memory-api directly - no duplication.

Usage:
    from agent.memory import DirectMemoryClient, MemoryConfig
    
    config = MemoryConfig()
    client = DirectMemoryClient(config)
    await client.initialize()
    
    # Recall - returns actual RecallResult from memory-api
    result = await client.recall("How to fix ImportError?")
    for fact in result.results:
        print(f"[{fact.fact_type}] {fact.text}")
    
    # Retain
    await client.retain(
        "Fixed ImportError by running pip install",
        context="error_fix"
    )
"""

from .config import MemoryConfig
from .base import BaseMemoryClient, NoOpMemoryClient
from .direct import DirectMemoryClient, get_memory_client
from .utils import (
    format_facts_for_prompt,
    build_recall_query,
    should_retain,
    extract_learning,
    get_error_from_output,
)
from .async_utils import run_async, close_async_runner


__all__ = [
    # Config
    "MemoryConfig",
    # Base classes
    "BaseMemoryClient",
    "NoOpMemoryClient",
    # Implementations
    "DirectMemoryClient",
    "get_memory_client",
    # Utilities
    "format_facts_for_prompt",
    "build_recall_query",
    "should_retain",
    "extract_learning",
    "get_error_from_output",
    # Async helpers
    "run_async",
    "close_async_runner",
]

