# from __future__ import annotations
from typing import Any, Optional, Union
import torch
import hydra
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from omegaconf import OmegaConf


class LlamaLitModel(torch.nn.Module):
    """A torch.nn.Module wrapper around a HuggingFace causal LM."""

    def __init__(
        self, config: Optional[Union[OmegaConf, Any]] = None,
        pretrained_model_name_or_path: Optional[str] = None, trust_remote_code: bool = False, torch_dtype: Optional[str] = None,
        use_cache: bool = False, gradient_checkpointing: bool = False, **kwargs: Any) -> None:

        super().__init__()

        # Resolve dtype string 
        dtype = None
        if isinstance(torch_dtype, str) and hasattr(torch, torch_dtype):
            dtype = getattr(torch, torch_dtype)

        if pretrained_model_name_or_path:
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=dtype)
        else:
            if config is None:
                raise TypeError("LlamaLitModel requires either config= or pretrained_model_name_or_path=")

            if OmegaConf.is_config(config):
                config = hydra.utils.instantiate(config)

            if hasattr(config, "use_cache"):
                config.use_cache = bool(use_cache)

            self.hf_model = LlamaForCausalLM(config)

        if gradient_checkpointing and hasattr(self.hf_model, "gradient_checkpointing_enable"):
            self.hf_model.gradient_checkpointing_enable()
            if hasattr(self.hf_model.config, "use_cache"):
                self.hf_model.config.use_cache = False

        # Avoid warnings and some edge case failures
        if getattr(self.hf_model.config, "pad_token_id", None) is None:
            self.hf_model.config.pad_token_id = getattr(self.hf_model.config, "eos_token_id", 0)

    @property
    def config(self):
        return self.hf_model.config

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs: Any):
        # config mismatch check
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        vocab_size = getattr(self.hf_model.config, "vocab_size", None)
        if vocab_size is not None:
            max_id = int(input_ids.max().item())
            min_id = int(input_ids.min().item())
            if min_id < 0 or max_id >= int(vocab_size):
                raise ValueError(
                    f"Token id out of range for vocab_size={vocab_size}: "
                    f"min={min_id}, max={max_id}. "
                    f"Fix by increasing config.vocab_size or changing dataset/tokenizer.")

        return self.hf_model(input_ids=input_ids, labels=labels, **kwargs)
