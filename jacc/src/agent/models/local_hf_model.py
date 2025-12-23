"""Local HuggingFace Model Provider with Optimization Techniques.

Load and run models locally using HuggingFace Transformers with various
optimization techniques to improve inference speed and reduce memory usage.

Optimization Techniques Available:
1. Flash Attention 2     - 2-4x faster, less memory (Ampere+ GPUs)
2. torch.compile         - 10-30% faster (PyTorch 2.0+)
3. BetterTransformer     - Native SDPA speedup
4. 8-bit Quantization    - 50% memory reduction (bitsandbytes)
5. 4-bit Quantization    - 75% memory reduction (bitsandbytes)
6. bfloat16/float16      - 50% memory, faster compute

Usage:
    model = LocalHFModel(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        use_flash_attention=True,
        use_torch_compile=True,
        quantization="4bit",
        torch_dtype="bfloat16",
    )
    response = model.query([{"role": "user", "content": "Hello!"}])
"""

import logging
import os
from typing import Any, Literal

import torch
from pydantic import BaseModel as PydanticBaseModel, Field

from .base import BaseModelProvider

logger = logging.getLogger(__name__)


class LocalHFModelConfig(PydanticBaseModel):
    """Configuration for local HuggingFace models."""
    
    model_name: str = Field(..., description="HuggingFace model ID or local path")
    
    # Generation parameters
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, gt=0, alias="max_new_tokens")
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1)
    repetition_penalty: float = Field(1.0, ge=1.0)
    do_sample: bool = Field(True)
    
    # Model loading options
    device_map: str | dict = Field("auto", description="Device placement strategy")
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field("auto")
    trust_remote_code: bool = Field(True)
    
    # Optimization flags
    use_flash_attention: bool = Field(True, description="Use Flash Attention 2 if available")
    use_torch_compile: bool = Field(False, description="Use torch.compile for speedup")
    use_better_transformer: bool = Field(False, description="Use BetterTransformer/SDPA")
    
    # Quantization
    quantization: Literal["4bit", "8bit", None] = Field(None, description="Quantization mode")
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16"] = Field("bfloat16")
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = Field("nf4")
    bnb_4bit_use_double_quant: bool = Field(True)
    
    # Memory optimization
    low_cpu_mem_usage: bool = Field(True)
    offload_folder: str | None = Field(None, description="Folder for CPU offloading")
    
    # Cache
    use_cache: bool = Field(True, description="Use KV cache during generation")
    
    # HuggingFace token
    hf_token: str | None = Field(None, description="HuggingFace token for gated models")


class LocalHFModel(BaseModelProvider):
    """Local HuggingFace Model with optimization techniques.
    
    Automatically applies the best available optimizations based on
    hardware capabilities and configuration.
    """
    
    def __init__(self, **kwargs):
        # Handle alias
        if "max_new_tokens" in kwargs and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
            
        self.config = LocalHFModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        
        self._model = None
        self._tokenizer = None
        self._device = None
        
        # Check for GPU
        self._has_cuda = torch.cuda.is_available()
        self._gpu_name = torch.cuda.get_device_name(0) if self._has_cuda else None
        
        # Initialize model and tokenizer
        self._load_model()
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype based on config and hardware."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        if self.config.torch_dtype == "auto":
            # Use bfloat16 on Ampere+ GPUs, float16 otherwise
            if self._has_cuda and self._gpu_name:
                ampere_or_newer = any(x in self._gpu_name for x in ["A100", "A10", "A6000", "RTX 30", "RTX 40", "H100"])
                return torch.bfloat16 if ampere_or_newer else torch.float16
            return torch.float32
        
        return dtype_map.get(self.config.torch_dtype, torch.float32)
    
    def _get_quantization_config(self):
        """Get BitsAndBytes quantization config if enabled."""
        if not self.config.quantization:
            return None
        
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.warning("bitsandbytes not installed, skipping quantization")
            return None
        
        if self.config.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif self.config.quantization == "4bit":
            compute_dtype = torch.bfloat16 if self.config.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )
        
        return None
    
    def _load_model(self) -> None:
        """Load model with optimizations."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Get token
        token = self.config.hf_token or os.getenv("HF_TOKEN")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            token=token,
        )
        
        # Ensure pad token exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Build model kwargs
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": self.config.device_map,
            "torch_dtype": self._get_torch_dtype(),
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "token": token,
        }
        
        # Add quantization config
        quant_config = self._get_quantization_config()
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
            logger.info(f"Using {self.config.quantization} quantization")
        
        # Flash Attention 2
        if self.config.use_flash_attention and self._has_cuda:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available: {e}")
        
        # Offload folder for CPU offloading
        if self.config.offload_folder:
            model_kwargs["offload_folder"] = self.config.offload_folder
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Apply BetterTransformer if requested
        if self.config.use_better_transformer:
            try:
                self._model = self._model.to_bettertransformer()
                logger.info("Using BetterTransformer")
            except Exception as e:
                logger.warning(f"BetterTransformer not available: {e}")
        
        # Apply torch.compile if requested
        if self.config.use_torch_compile:
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead")
                logger.info("Using torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")
        
        # Set model to eval mode
        self._model.eval()
        
        logger.info(f"Model loaded successfully on {self.config.device_map}")
    
    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages using model's chat template."""
        if hasattr(self._tokenizer, 'apply_chat_template'):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Fallback format
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
    
    @torch.inference_mode()
    def _generate(self, prompt: str, **kwargs) -> tuple[str, dict]:
        """Generate response from prompt."""
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._model.device)
        
        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
            "use_cache": self.config.use_cache,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        
        # Handle temperature=0 case
        if gen_kwargs["temperature"] == 0:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature")
            gen_kwargs.pop("top_p")
            gen_kwargs.pop("top_k")
        
        # Generate
        input_length = inputs.input_ids.shape[1]
        outputs = self._model.generate(**inputs, **gen_kwargs)
        
        # Decode only new tokens
        generated_ids = outputs[0][input_length:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        usage = {
            "prompt_tokens": input_length,
            "completion_tokens": len(generated_ids),
            "total_tokens": input_length + len(generated_ids),
        }
        
        return generated_text, usage
    
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Query the local model."""
        normalized = [{"role": m["role"], "content": m["content"]} for m in messages]
        prompt = self._format_messages(normalized)
        
        generated_text, usage = self._generate(prompt, **kwargs)
        
        self._track_cost(0.0)  # Local models have no API cost
        
        return {
            "content": generated_text.strip(),
            "extra": {
                "usage": usage,
                "model": self.config.model_name,
                "optimizations": self._get_applied_optimizations(),
            },
        }
    
    def _get_applied_optimizations(self) -> list[str]:
        """Get list of applied optimizations."""
        opts = []
        if self.config.use_flash_attention:
            opts.append("flash_attention_2")
        if self.config.use_torch_compile:
            opts.append("torch_compile")
        if self.config.use_better_transformer:
            opts.append("better_transformer")
        if self.config.quantization:
            opts.append(f"quantization_{self.config.quantization}")
        if self.config.torch_dtype != "float32":
            opts.append(f"dtype_{self.config.torch_dtype}")
        return opts
    
    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables."""
        return {
            **self.config.model_dump(),
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "provider": "local_hf",
            "optimizations": self._get_applied_optimizations(),
            "gpu": self._gpu_name,
        }
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        if self._has_cuda:
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded")
