"""Model Providers Package.

Unified interface for multiple model backends:

1. APIModel         - API providers via LiteLLM (OpenAI, Gemini, Anthropic, Deepseek...)
2. VLLMModel        - vLLM server or offline mode
3. HFInferenceModel - HuggingFace Serverless Inference API
4. LocalHFModel     - Local HuggingFace with optimizations (Flash Attention, quantization...)

Usage:
    from agent.model import get_model
    
    # API model (default)
    model = get_model("api", model_name="gemini/gemini-2.0-flash")
    
    # vLLM server
    model = get_model("vllm", model_name="meta-llama/Llama-3.1-8B-Instruct", mode="server")
    
    # HuggingFace Inference API  
    model = get_model("hf_inference", model_name="meta-llama/Llama-3.1-8B-Instruct")
    
    # Local HuggingFace with optimizations
    model = get_model(
        "local_hf",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        quantization="4bit",
        use_flash_attention=True
    )
    
    # Query
    response = model.query([{"role": "user", "content": "Hello!"}])
    print(response["content"])
"""

from typing import Literal

from .base import BaseModelProvider, BaseModelConfig
from .api_model import APIModel, APIModelConfig
from .vllm_model import VLLMModel, VLLMModelConfig
from .hf_inference_model import HFInferenceModel, HFInferenceModelConfig
from .local_hf_model import LocalHFModel, LocalHFModelConfig

# Type alias for provider names
ProviderType = Literal["api", "vllm", "hf_inference", "local_hf"]

# Provider mapping
_PROVIDER_MAP = {
    "api": APIModel,
    "vllm": VLLMModel,
    "hf_inference": HFInferenceModel,
    "local_hf": LocalHFModel,
}


def get_model(provider: ProviderType = "api", **kwargs) -> BaseModelProvider:
    """Create a model instance based on provider type.
    
    Args:
        provider: One of "api", "vllm", "hf_inference", "local_hf"
        **kwargs: Configuration passed to the model constructor
        
    Returns:
        Configured model instance
        
    Examples:
        >>> model = get_model("api", model_name="gpt-4o-mini")
        >>> model = get_model("vllm", model_name="llama", mode="server", server_url="http://localhost:8000")
        >>> model = get_model("local_hf", model_name="meta-llama/Llama-3.1-8B-Instruct", quantization="4bit")
    """
    if provider not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available providers: {list(_PROVIDER_MAP.keys())}"
        )
    
    model_class = _PROVIDER_MAP[provider]
    return model_class(**kwargs)


# Convenience functions for each provider
def api_model(model_name: str, **kwargs) -> APIModel:
    """Create an API model (LiteLLM)."""
    return APIModel(model_name=model_name, **kwargs)


def vllm_model(model_name: str, **kwargs) -> VLLMModel:
    """Create a vLLM model."""
    return VLLMModel(model_name=model_name, **kwargs)


def hf_inference_model(model_name: str, **kwargs) -> HFInferenceModel:
    """Create a HuggingFace Inference API model."""
    return HFInferenceModel(model_name=model_name, **kwargs)


def local_hf_model(model_name: str, **kwargs) -> LocalHFModel:
    """Create a local HuggingFace model."""
    return LocalHFModel(model_name=model_name, **kwargs)


__all__ = [
    # Base
    "BaseModelProvider",
    "BaseModelConfig",
    
    # Providers
    "APIModel",
    "APIModelConfig",
    "VLLMModel",
    "VLLMModelConfig",
    "HFInferenceModel",
    "HFInferenceModelConfig",
    "LocalHFModel",
    "LocalHFModelConfig",
    
    # Factory
    "get_model",
    "ProviderType",
    
    # Convenience
    "api_model",
    "vllm_model",
    "hf_inference_model",
    "local_hf_model",
]
