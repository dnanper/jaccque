"""API Model Provider using LiteLLM.

Supports all major API providers through LiteLLM unified interface.

Usage:
    model = APIModel(model_name="gemini/gemini-2.0-flash")
    response = model.query([{"role": "user", "content": "Hello!"}])
"""

import logging
import os
from typing import Any, Literal

import litellm
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseModelProvider

logger = logging.getLogger(__name__)


class APIModelConfig(BaseModel):
    """Configuration for API-based models."""
    model_name: str = Field(..., description="Model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(4096, gt=0, description="Maximum tokens in response")
    
    # LiteLLM specific
    api_key: str | None = Field(None, description="API key (or use env vars)")
    api_base: str | None = Field(None, description="Custom API base URL")
    
    # Additional generation parameters
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    
    # Retry and tracking settings
    max_retries: int = Field(10, ge=1, description="Maximum retry attempts")
    cost_tracking: Literal["default", "ignore_errors"] = Field(
        "ignore_errors", 
        description="Cost tracking mode"
    )
    
    # Streaming
    stream: bool = Field(False, description="Enable streaming mode")

class APIModel(BaseModelProvider):
    """API Model using LiteLLM as unified interface.
    
    Environment variables for API keys:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY  
    - GEMINI_API_KEY (or GOOGLE_API_KEY)
    - DEEPSEEK_API_KEY
    """
    
    def __init__(self, **kwargs):
        self.config = APIModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        
        # Set API key if provided
        if self.config.api_key:
            # Determine provider from model name and set appropriate env var
            self._set_api_key(self.config.api_key)
    
    # Auto set api key in env
    def _set_api_key(self, api_key: str) -> None:
        """Set API key based on model provider."""
        model_lower = self.config.model_name.lower()
        
        if "gemini" in model_lower or "google" in model_lower:
            os.environ["GEMINI_API_KEY"] = api_key
        elif "claude" in model_lower or "anthropic" in model_lower:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif "deepseek" in model_lower:
            os.environ["DEEPSEEK_API_KEY"] = api_key
        else:
            # Default to OpenAI format
            os.environ["OPENAI_API_KEY"] = api_key
    
    def _build_model_kwargs(self, **kwargs) -> dict[str, Any]:
        """Build kwargs for litellm.completion call."""
        model_kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        # Add optional parameters if set
        if self.config.top_p is not None:
            model_kwargs["top_p"] = self.config.top_p
        if self.config.frequency_penalty is not None:
            model_kwargs["frequency_penalty"] = self.config.frequency_penalty
        if self.config.presence_penalty is not None:
            model_kwargs["presence_penalty"] = self.config.presence_penalty
        if self.config.api_base:
            model_kwargs["api_base"] = self.config.api_base
        if self.config.stream:
            model_kwargs["stream"] = True
            
        # Override with any passed kwargs
        model_kwargs.update(kwargs)
        
        return model_kwargs
    
    @retry(
        reraise=True,
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((
            litellm.exceptions.AuthenticationError,
            litellm.exceptions.NotFoundError,
            litellm.exceptions.PermissionDeniedError,
            KeyboardInterrupt,
        )),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs) -> Any:
        """Internal query with retry logic."""
        model_kwargs = self._build_model_kwargs(**kwargs)
        
        try:
            response = litellm.completion(
                model=self.config.model_name,
                messages=messages,
                **model_kwargs
            )
            return response
        except litellm.exceptions.AuthenticationError as e:
            logger.error(f"Authentication failed for {self.config.model_name}")
            raise
    
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Query the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters passed to litellm
            
        Returns:
            dict with 'content' and 'extra' keys
        """
        # Normalize messages
        # Chat template, include all messages happened in conversation (query, response, observation)
        normalized_messages = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in messages
        ]
        
        response = self._query(normalized_messages, **kwargs)
        
        # Handle streaming response
        if self.config.stream:
            return self._handle_stream(response)
        
        # Calculate cost
        cost = self._calculate_cost(response)
        self._track_cost(cost)
        
        return {
            "content": response.choices[0].message.content or "",
            "extra": {
                "response": response.model_dump() if hasattr(response, 'model_dump') else dict(response),
                "cost": cost,
                "usage": response.usage.model_dump() if response.usage else {},
            },
        }
    
    def _handle_stream(self, response) -> dict[str, Any]:
        """Handle streaming response."""
        content_parts = []
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)
        
        full_content = "".join(content_parts)
        self._track_cost(0.0)  # Cost tracking for streaming is limited
        
        return {
            "content": full_content,
            "extra": {"streamed": True},
        }
    
    def _calculate_cost(self, response) -> float:
        """Calculate cost using litellm's cost calculator."""
        try:
            cost = litellm.cost_calculator.completion_cost(
                response, 
                model=self.config.model_name
            )
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
            return cost
        except Exception as e:
            if self.config.cost_tracking != "ignore_errors":
                logger.warning(f"Error calculating cost: {e}")
            return 0.0
    
    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables."""
        return {
            **self.config.model_dump(),
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "provider": "api",
        }
    
    # Convenience class methods for common providers
    @classmethod
    def gemini(cls, model: str = "gemini-2.0-flash", **kwargs) -> "APIModel":
        """Create a Gemini model."""
        return cls(model_name=f"gemini/{model}", **kwargs)
    
    @classmethod
    def openai(cls, model: str = "gpt-4o-mini", **kwargs) -> "APIModel":
        """Create an OpenAI model."""
        return cls(model_name=model, **kwargs)
    
    @classmethod
    def anthropic(cls, model: str = "claude-3-5-sonnet-20241022", **kwargs) -> "APIModel":
        """Create an Anthropic model."""
        return cls(model_name=model, **kwargs)
    
    @classmethod
    def deepseek(cls, model: str = "deepseek-chat", **kwargs) -> "APIModel":
        """Create a Deepseek model."""
        return cls(model_name=f"deepseek/{model}", **kwargs)
