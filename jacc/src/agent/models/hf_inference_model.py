"""HuggingFace Inference API Model Provider.

Uses HuggingFace's Serverless Inference API to run models hosted on HF infrastructure.
Supports both free tier (rate-limited) and Inference Endpoints (dedicated).

Usage:
    model = HFInferenceModel(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        hf_token="hf_..."  # or set HF_TOKEN env var
    )
    response = model.query([{"role": "user", "content": "Hello!"}])
"""

import json
import logging
import os
from typing import Any

import requests
from pydantic import BaseModel as PydanticBaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseModelProvider

logger = logging.getLogger(__name__)


class HFInferenceModelConfig(PydanticBaseModel):
    """Configuration for HuggingFace Inference API."""
    
    model_name: str = Field(..., description="HuggingFace model ID")
    hf_token: str | None = Field(None, description="HuggingFace token")
    
    # Endpoint configuration
    endpoint_url: str | None = Field(None, description="Custom Inference Endpoint URL")
    
    # Generation parameters
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, gt=0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1)
    repetition_penalty: float = Field(1.0, ge=1.0)
    do_sample: bool = Field(True)
    stop: list[str] | None = Field(None)
    
    # Request settings
    timeout: int = Field(120)
    max_retries: int = Field(5)
    use_chat_api: bool = Field(True, description="Use chat completions format")


class HFInferenceAPIError(Exception):
    """HuggingFace Inference API error."""
    pass


class HFRateLimitError(Exception):
    """Rate limit exceeded."""
    pass


class HFInferenceModel(BaseModelProvider):
    """HuggingFace Inference API Model Provider."""
    
    INFERENCE_API_URL = "https://api-inference.huggingface.co/models"
    CHAT_API_URL = "https://api-inference.huggingface.co/v1/chat/completions"
    
    def __init__(self, **kwargs):
        self.config = HFInferenceModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self._token = self.config.hf_token or os.getenv("HF_TOKEN")
        
        if not self._token:
            logger.warning("No HF_TOKEN found. Some models may not be accessible.")
    
    @property
    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers
    
    @property
    def _api_url(self) -> str:
        """Get API URL based on configuration."""
        if self.config.endpoint_url:
            return self.config.endpoint_url
        if self.config.use_chat_api:
            return self.CHAT_API_URL
        return f"{self.INFERENCE_API_URL}/{self.config.model_name}"
    
    def _build_chat_payload(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Build payload for chat completions API."""
        return {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }
    
    def _build_text_gen_payload(self, prompt: str, **kwargs) -> dict:
        """Build payload for text generation API."""
        parameters = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty,
            "return_full_text": False,
        }
        if self.config.top_k:
            parameters["top_k"] = self.config.top_k
        if self.config.stop:
            parameters["stop_sequences"] = self.config.stop
        return {"inputs": prompt, "parameters": parameters}
    
    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert messages to prompt format."""
        parts = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant:")
        return "".join(parts)
    
    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_exception_type(HFRateLimitError),
    )
    def _request(self, payload: dict) -> dict:
        """Make API request with retry logic."""
        response = requests.post(
            self._api_url,
            headers=self._headers,
            json=payload,
            timeout=self.config.timeout,
        )
        
        if response.status_code == 429:
            raise HFRateLimitError("Rate limit exceeded")
        
        if response.status_code == 503:
            # Model loading
            data = response.json()
            if "estimated_time" in data:
                logger.info(f"Model loading, estimated time: {data['estimated_time']}s")
            raise HFRateLimitError("Model is loading")
        
        if not response.ok:
            raise HFInferenceAPIError(f"API error {response.status_code}: {response.text}")
        
        return response.json()
    
    def _query_chat_api(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Query using chat completions API."""
        payload = self._build_chat_payload(messages, **kwargs)
        response = self._request(payload)
        
        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        
        return {
            "content": content,
            "extra": {
                "usage": usage,
                "model": response.get("model"),
                "finish_reason": response["choices"][0].get("finish_reason"),
            },
        }
    
    def _query_text_gen_api(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Query using text generation API."""
        prompt = self._messages_to_prompt(messages)
        payload = self._build_text_gen_payload(prompt, **kwargs)
        response = self._request(payload)
        
        # Response format varies
        if isinstance(response, list):
            content = response[0].get("generated_text", "")
        else:
            content = response.get("generated_text", "")
        
        return {
            "content": content,
            "extra": {"raw_response": response},
        }
    
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Query the HuggingFace Inference API."""
        normalized = [{"role": m["role"], "content": m["content"]} for m in messages]
        
        if self.config.use_chat_api:
            result = self._query_chat_api(normalized, **kwargs)
        else:
            result = self._query_text_gen_api(normalized, **kwargs)
        
        self._track_cost(0.0)  # HF Inference API doesn't report cost
        return result
    
    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables."""
        return {
            **self.config.model_dump(),
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "provider": "hf_inference",
        }
