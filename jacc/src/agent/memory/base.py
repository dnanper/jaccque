"""Abstract interface for Memory clients.

For Direct client, we import and use classes directly from memory-api.
This keeps things simple and avoids duplication.
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    # These are the actual classes from memory-api
    from src.engine.response_models import RecallResult, MemoryFact


class BaseMemoryClient(ABC):
    """Abstract interface for memory clients.
    
    Implementations:
    - DirectMemoryClient: Calls MemoryEngine directly (dev)
    - HTTPMemoryClient: Calls Memory API via HTTP (production) - future
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the client (connect to database, etc.)."""
        ...
    
    @abstractmethod
    async def recall(
        self,
        query: str,
        *,
        max_tokens: int = 2048,
        budget: str = "mid",
        fact_types: list[str] | None = None,
        include_entities: bool = False,
    ) -> "RecallResult":
        """Recall relevant memories for a query.
        
        Returns the actual RecallResult from memory-api.
        """
        ...
    
    @abstractmethod
    async def retain(
        self,
        content: str,
        *,
        context: str | None = None,
        metadata: dict[str, str] | None = None,
        fact_type_override: str | None = None,
    ) -> list[str]:
        """Store a new memory.
        
        Returns list of unit IDs.
        """
        ...
    
    @abstractmethod
    async def retain_batch(
        self,
        contents: list[dict[str, Any]],
    ) -> list[list[str]]:
        """Store multiple memories in batch.
        
        Returns list of unit ID lists.
        """
        ...
    
    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        ...
    
    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        ...


class NoOpMemoryClient(BaseMemoryClient):
    """No-op memory client for when memory is disabled.
    
    All operations succeed but do nothing.
    """
    
    def __init__(self):
        self._initialized = True
    
    async def initialize(self) -> None:
        pass
    
    async def recall(self, query: str, **kwargs):
        # Return a minimal RecallResult-like object
        # Must match memory-api's RecallResult: has 'results' not 'facts'
        class EmptyRecallResult:
            results = []  # Same attribute name as memory-api RecallResult
            trace = None
            entities = None
            chunks = None
        return EmptyRecallResult()
    
    async def retain(self, content: str, **kwargs) -> list[str]:
        return []
    
    async def retain_batch(self, contents: list[dict[str, Any]]) -> list[list[str]]:
        return [[] for _ in contents]
    
    async def close(self) -> None:
        pass
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized


__all__ = [
    "BaseMemoryClient",
    "NoOpMemoryClient",
]
