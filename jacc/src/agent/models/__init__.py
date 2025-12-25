"""Model Providers Package.

Unified interface for multiple model backends:

1. APIModel         - API providers via LiteLLM (OpenAI, Gemini, Anthropic, Deepseek...)
2. VLLMModel        - vLLM server or offline mode
3. HFInferenceModel - HuggingFace Serverless Inference API
4. LocalHFModel     - Local HuggingFace with optimizations (Flash Attention, quantization...)

Usage:
    from agent.models import get_model
    
    # API model (default) - lightweight, no heavy dependencies
    model = get_model("api", model_name="gemini/gemini-2.0-flash")
    
    # Query
    response = model.query([{"role": "user", "content": "Hello!"}])
    print(response["content"])
"""

from typing import Literal, TYPE_CHECKING

from .base import BaseModelProvider, BaseModelConfig

# Only import lightweight models at module level
from .api_model import APIModel, APIModelConfig


# Type alias for provider names
ProviderType = Literal["api", "vllm", "hf_inference", "local_hf"]


def get_model(provider: ProviderType = "api", **kwargs) -> BaseModelProvider:
    """Create a model instance based on provider type.
    
    Lazy imports heavy dependencies (torch, transformers) only when needed.
    
    Args:
        provider: One of "api", "vllm", "hf_inference", "local_hf"
        **kwargs: Configuration passed to the model constructor
        
    Returns:
        Configured model instance
    """
    if provider == "api":
        return APIModel(**kwargs)
    
    elif provider == "vllm":
        from .vllm_model import VLLMModel
        return VLLMModel(**kwargs)
    
    elif provider == "hf_inference":
        from .hf_inference_model import HFInferenceModel
        return HFInferenceModel(**kwargs)
    
    elif provider == "local_hf":
        from .local_hf_model import LocalHFModel
        return LocalHFModel(**kwargs)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: api, vllm, hf_inference, local_hf"
        )


# Convenience function for API (most common)
def api_model(model_name: str, **kwargs) -> APIModel:
    """Create an API model (LiteLLM)."""
    return APIModel(model_name=model_name, **kwargs)


__all__ = [
    # Base
    "BaseModelProvider",
    "BaseModelConfig",
    
    # API (always available)
    "APIModel",
    "APIModelConfig",
    
    # Factory
    "get_model",
    "ProviderType",
    
    # Convenience
    "api_model",
]
