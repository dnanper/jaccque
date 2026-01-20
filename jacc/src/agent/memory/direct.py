"""Direct Memory Client - Calls MemoryEngine directly.

This implementation imports and uses classes directly from memory-api.
No duplication - we just wrap the MemoryEngine for agent use.

Usage:
    from agent.memory import DirectMemoryClient, MemoryConfig
    
    config = MemoryConfig()
    client = DirectMemoryClient(config)
    await client.initialize()
    
    # Recall - returns actual RecallResult from memory-api
    result = await client.recall("How to fix ImportError?")
    for fact in result.results:
        print(f"[{fact.fact_type}] {fact.text}")
"""

import logging
import sys
from pathlib import Path
from typing import Any

from .base import BaseMemoryClient
from .config import MemoryConfig


logger = logging.getLogger(__name__)


# Add memory-api to path for imports
def _setup_import_path():
    """Add memory-api source to Python path."""
    agent_dir = Path(__file__).parent.parent  # agent/
    jacc_src = agent_dir.parent  # jacc/src/
    memory_api_src = jacc_src / "memory-api" / "api"
    
    if memory_api_src.exists():
        src_path = str(memory_api_src)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            logger.debug(f"Added to path: {src_path}")

_setup_import_path()


class DirectMemoryClient(BaseMemoryClient):
    """Memory client that calls MemoryEngine directly.
    
    For development use - requires only database container.
    Uses actual classes from memory-api, no duplication.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize the direct memory client.
        
        Args:
            config: Memory configuration
        """
        self.config = config
        self._engine = None
        self._initialized = False
        self._RequestContext = None
        self._Budget = None
    
    async def initialize(self) -> None:
        """Initialize the MemoryEngine (connect to database, load models)."""
        if self._initialized:
            return
        
        try:
            # Import directly from memory-api
            from src import MemoryEngine
            from src.models import RequestContext
            from src.engine.memory_engine import Budget
            
            # Store for later use
            self._RequestContext = RequestContext
            self._Budget = Budget
            
            logger.info(f"Initializing MemoryEngine with db: {self.config.database_url[:50]}...")
            
            # Create engine with config
            self._engine = MemoryEngine(
                db_url=self.config.database_url,
                run_migrations=True,
            )
            
            # Initialize
            await self._engine.initialize()
            
            self._initialized = True
            logger.info("MemoryEngine initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import MemoryEngine: {e}")
            logger.error("Make sure memory-api is in PYTHONPATH or path setup is correct")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MemoryEngine: {e}")
            raise
    
    async def recall(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        budget: str | None = None,
        fact_types: list[str] | None = None,
        include_entities: bool | None = None,
    ):
        """Recall relevant memories for a query.
        
        Returns the actual RecallResult from memory-api.
        """
        if not self._initialized:
            await self.initialize()
        
        # Apply defaults from config
        max_tokens = max_tokens or self.config.max_recall_tokens
        budget = budget or self.config.recall_budget
        fact_types = fact_types or self.config.fact_types
        include_entities = include_entities if include_entities is not None else self.config.include_entities
        
        try:
            # Map string budget to enum
            budget_map = {
                "low": self._Budget.LOW,
                "mid": self._Budget.MID,
                "high": self._Budget.HIGH,
            }
            budget_enum = budget_map.get(budget.lower(), self._Budget.MID)
            
            # Create request context
            ctx = self._RequestContext()
            
            # Call recall - returns actual RecallResult from memory-api
            result = await self._engine.recall_async(
                bank_id=self.config.bank_id,
                query=query,
                budget=budget_enum,
                max_tokens=max_tokens,
                fact_type=fact_types,
                include_entities=include_entities,
                request_context=ctx,
            )
            
            logger.debug(f"Recalled {len(result.results)} facts for query: {query[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            # Return empty result on error
            from .base import NoOpMemoryClient
            return await NoOpMemoryClient().recall(query)
    
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
        if not self.config.enable_retain:
            return []
        
        if not self._initialized:
            await self.initialize()
        
        try:
            ctx = self._RequestContext()
            
            # Build content dict
            content_dict: dict[str, Any] = {
                "content": content,
            }
            if context:
                content_dict["context"] = context
            if metadata:
                content_dict["metadata"] = metadata
            
            # Call retain
            result = await self._engine.retain_batch_async(
                bank_id=self.config.bank_id,
                contents=[content_dict],
                request_context=ctx,
                fact_type_override=fact_type_override,
            )
            
            # Flatten result
            unit_ids = []
            for id_list in result:
                unit_ids.extend(id_list)
            
            logger.debug(f"Retained {len(unit_ids)} facts: {content[:50]}...")
            
            return unit_ids
            
        except Exception as e:
            logger.error(f"Retain failed: {e}")
            return []
    
    async def retain_batch(
        self,
        contents: list[dict[str, Any]],
    ) -> list[list[str]]:
        """Store multiple memories in batch.
        
        Returns list of unit ID lists.
        """
        if not self.config.enable_retain:
            return [[] for _ in contents]
        
        if not self._initialized:
            await self.initialize()
        
        try:
            ctx = self._RequestContext()
            
            result = await self._engine.retain_batch_async(
                bank_id=self.config.bank_id,
                contents=contents,
                request_context=ctx,
            )
            
            logger.debug(f"Batch retained {len(result)} items")
            
            return [list(ids) for ids in result]
            
        except Exception as e:
            logger.error(f"Batch retain failed: {e}")
            return [[] for _ in contents]
    
    async def close(self) -> None:
        """Close the engine and release resources."""
        if self._engine:
            await self._engine.close()
            self._engine = None
        self._initialized = False
        logger.info("MemoryEngine closed")
    
    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized


def get_memory_client(
    config: MemoryConfig | None = None,
    mode: str = "direct",
) -> BaseMemoryClient:
    """Factory function to create a memory client.
    
    Args:
        config: Memory configuration (uses defaults if None)
        mode: Client mode - 'direct' or 'noop'
        
    Returns:
        Memory client instance
    """
    from .base import NoOpMemoryClient
    
    if mode == "noop":
        return NoOpMemoryClient()
    
    config = config or MemoryConfig.from_env()
    
    if mode == "direct":
        return DirectMemoryClient(config)
    
    raise ValueError(f"Unknown memory client mode: {mode}")


__all__ = ["DirectMemoryClient", "get_memory_client"]
