"""Memory configuration for agent integration.

Defines configuration for connecting to the Memory system.
Uses environment variables with sensible defaults for development.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """Configuration for Memory client.
    
    Attributes:
        database_url: PostgreSQL connection URL (from MEMORY_API_DATABASE_URL)
        bank_id: Memory bank ID to use (can be per-instance or shared)
        max_recall_tokens: Max tokens to retrieve in recall (default: 2048)
        recall_budget: Budget level for recall ('low', 'mid', 'high')
        fact_types: Which fact types to recall
        enable_retain: Whether to store new learnings
        retain_on_error_fix: Retain when error is fixed
        retain_on_task_complete: Retain task summary on completion
    """
    
    # Connection
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "MEMORY_API_DATABASE_URL",
            "postgresql://memory:memory_secret@localhost:5432/memory_db"
        ),
        description="PostgreSQL connection URL"
    )
    
    # Bank configuration
    bank_id: str = Field(
        default="swe_agent",
        description="Memory bank ID (shared across tasks for experience transfer)"
    )
    
    # Recall settings
    max_recall_tokens: int = Field(default=2048, gt=0)
    recall_budget: str = Field(default="mid", pattern="^(low|mid|high)$")
    fact_types: list[str] = Field(
        default_factory=lambda: ["world", "experience", "opinion"],
        description="Which fact types to recall"
    )
    include_entities: bool = Field(default=True)
    
    # Retain settings
    enable_retain: bool = Field(default=True)
    retain_on_error_fix: bool = Field(default=True)
    retain_on_task_complete: bool = Field(default=True)
    
    # Context settings
    include_repo_context: bool = Field(
        default=True,
        description="Include repo name in retain metadata"
    )
    
    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create config from environment variables."""
        return cls(
            database_url=os.getenv(
                "MEMORY_API_DATABASE_URL",
                "postgresql://memory:memory_secret@localhost:5432/memory_db"
            ),
            bank_id=os.getenv("MEMORY_BANK_ID", "swe_agent"),
            max_recall_tokens=int(os.getenv("MEMORY_MAX_RECALL_TOKENS", "2048")),
            recall_budget=os.getenv("MEMORY_RECALL_BUDGET", "mid"),
            enable_retain=os.getenv("MEMORY_ENABLE_RETAIN", "true").lower() == "true",
        )
    
    @classmethod
    def for_instance(cls, instance_id: str, **kwargs) -> "MemoryConfig":
        """Create config for a specific SWE-bench instance.
        
        Uses shared bank by default for experience transfer.
        Pass bank_id in kwargs to use per-instance banks.
        """
        config = cls.from_env()
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


__all__ = ["MemoryConfig"]
