"""vLLM Model Provider.

Supports two modes:
1. Server Mode: Connect to a running vLLM server via OpenAI-compatible API
2. Offline Mode: Load model directly in process using vLLM engine

Usage (Server Mode):
    # First start vLLM server:
    # vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
    
    model = VLLMModel(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        mode="server",
        server_url="http://localhost:8000"
    )

Usage (Offline Mode):
    model = VLLMModel(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        mode="offline",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel as PydanticBaseModel, Field

from .base import BaseModelProvider

logger = logging.getLogger(__name__)


class VLLMModelConfig(PydanticBaseModel):
    """Configuration for vLLM models."""
    
    model_name: str = Field(..., description="HuggingFace model name or path")
    mode: Literal["server", "offline"] = Field("server", description="vLLM mode")
    
    # Generation parameters
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, gt=0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(-1, description="-1 means disabled")
    repetition_penalty: float = Field(1.0, ge=1.0)
    stop: list[str] | None = Field(None, description="Stop sequences")
    
    # Server mode settings
    server_url: str = Field("http://localhost:8000", description="vLLM server URL")
    api_key: str | None = Field(None, description="API key if server requires auth")
    
    # Offline mode settings
    tensor_parallel_size: int = Field(1, ge=1, description="Number of GPUs")
    gpu_memory_utilization: float = Field(0.9, ge=0.1, le=1.0)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field("auto")
    quantization: Literal["awq", "gptq", "squeezellm", None] = Field(None)
    max_model_len: int | None = Field(None, description="Maximum context length")
    trust_remote_code: bool = Field(True)
    
    # Speculative decoding
    speculative_model: str | None = Field(None, description="Draft model for speculative decoding")
    num_speculative_tokens: int = Field(5)


class VLLMModel(BaseModelProvider):
    """vLLM Model Provider with server and offline modes."""
    
    def __init__(self, **kwargs):
        self.config = VLLMModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        
        self._engine = None
        self._client = None
        self._tokenizer = None
        
        if self.config.mode == "offline":
            self._init_offline_engine()
        else:
            self._init_server_client()
    
    def _init_server_client(self) -> None:
        """Initialize OpenAI client for server mode."""
        try:
            from openai import OpenAI
            
            self._client = OpenAI(
                base_url=f"{self.config.server_url}/v1",    # Do not query to Openai Server, instead query to my vLLM server
                api_key=self.config.api_key or "dummy",      # Do not use Openai API key
            )
            logger.info(f"Connected to vLLM server at {self.config.server_url}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def _init_offline_engine(self) -> None:
        """Initialize vLLM engine for offline mode."""
        try:
            from vllm import LLM, SamplingParams
            
            engine_kwargs = {
                "model": self.config.model_name,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "dtype": self.config.dtype,
                "trust_remote_code": self.config.trust_remote_code,
            }
            
            if self.config.quantization:
                engine_kwargs["quantization"] = self.config.quantization
            if self.config.max_model_len:
                engine_kwargs["max_model_len"] = self.config.max_model_len
            if self.config.speculative_model:
                engine_kwargs["speculative_model"] = self.config.speculative_model
                engine_kwargs["num_speculative_tokens"] = self.config.num_speculative_tokens
            
            self._engine = LLM(**engine_kwargs)
            self._sampling_params_class = SamplingParams
            
            logger.info(f"Loaded vLLM engine: {self.config.model_name}")
        except ImportError:
            raise ImportError(
                "Please install vllm: pip install vllm\n"
                "Note: vLLM requires CUDA and a compatible GPU."
            )
    
    def _build_sampling_params(self, **kwargs) -> Any:
        """Build SamplingParams for offline mode."""
        params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
        }
        if self.config.stop:
            params["stop"] = self.config.stop
        return self._sampling_params_class(**params)
    
    def _format_chat_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages using model's chat template if available."""
        try:
            from transformers import AutoTokenizer
            
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
            
            # Use the model's chat template
            if hasattr(self._tokenizer, 'apply_chat_template'):
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception as e:
            logger.warning(f"Could not apply chat template: {e}")
        
        # Fallback to simple format, no chat template
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant:")
        return "".join(parts)
    
    def _query_server(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Query vLLM server using OpenAI-compatible API."""
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )
        
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        
        return {
            "content": content,
            "extra": {
                "usage": usage,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            },
        }
    
    def _query_offline(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Query vLLM engine directly."""
        prompt = self._format_chat_messages(messages)
        sampling_params = self._build_sampling_params(**kwargs)
        
        outputs = self._engine.generate([prompt], sampling_params)
        output = outputs[0]
        
        generated_text = output.outputs[0].text
        
        return {
            "content": generated_text,
            "extra": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason,
            },
        }
    
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Query the vLLM model."""
        normalized = [{"role": m["role"], "content": m["content"]} for m in messages]
        
        if self.config.mode == "server":
            result = self._query_server(normalized, **kwargs)
        else:
            result = self._query_offline(normalized, **kwargs)
        
        self._track_cost(0.0)  # Local models don't have API cost
        return result
    
    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables."""
        return {
            **self.config.model_dump(),
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "provider": "vllm",
        }
