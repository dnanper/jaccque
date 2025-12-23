"""Base model interface for all model providers."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel as PydanticBaseModel


class BaseModelConfig(PydanticBaseModel):
    """Base configuration for all models."""
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096


class BaseModelProvider(ABC):
    """Abstract base class for all model providers.
    
    All model providers must implement:
    - query(): Single query to the model
    - get_template_vars(): Return template variables for logging/debugging
    """
    
    config: BaseModelConfig
    cost: float = 0.0
    n_calls: int = 0
    
    @abstractmethod
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Query the model with a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional model-specific parameters
            
        Returns:
            dict with keys:
                - content: str - The model's response text
                - extra: dict - Additional metadata (usage, cost, etc.)
        """
        pass
    
    @abstractmethod
    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables for logging/debugging.
        
        Returns:
            dict with model configuration and statistics
        """
        pass
    
    def _track_cost(self, cost: float) -> None:
        """Track cost and call count."""
        self.cost += cost
        self.n_calls += 1