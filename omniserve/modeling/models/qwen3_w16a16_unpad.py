"""Qwen3 w16a16 model for LServe with unified sparse attention.

Qwen3 is structurally identical to Llama except for per-head Q/K RMSNorm
applied before RoPE.  We reuse all Llama infrastructure and only override
the attention module and the top-level CausalLM class.

Note: models with rope_scaling["type"] == "yarn" (e.g. Qwen3-30B-A3B) will
hit an assertion in ctx_update_kv.py; linear/None scaling works today.
"""
import torch
from torch import nn
from typing import Dict, List, Optional
from transformers import Qwen3Config

from omniserve.config import ModelConfig
from omniserve.modeling.layers.layernorm import RMSNorm
from omniserve.modeling.layers.sampler import Sampler
from omniserve.modeling.models.llama_w16a16_unpad import (
    LlamaAttention,
    LlamaMLP,
    LlamaModel,
    LlamaForCausalLM,
)
from omniserve.sampling_params import SamplingParams
from omniserve.utils.input_metadata import InputMetadata
from omniserve.utils.quant_config import QServeQuantConfig


class Qwen3Attention(LlamaAttention):
    """Adds per-head Q/K RMSNorm on top of LlamaAttention."""

    def __init__(self, args, model_config: ModelConfig, layer_idx: int,
                 kv_cache_config: Optional[Dict] = None) -> None:
        super().__init__(args, model_config, layer_idx, kv_cache_config)
        eps = getattr(args, "rms_norm_eps", 1e-6)
        self.q_norm = RMSNorm(self.head_dim, eps=eps, use_quant=False)
        self.k_norm = RMSNorm(self.head_dim, eps=eps, use_quant=False)

    def _apply_qk_norm(self, qkv: torch.Tensor) -> None:
        """In-place per-head Q/K normalization on the packed QKV buffer."""
        t = qkv.shape[0]
        qkv[:, :self.q_size] = self.q_norm(
            qkv[:, :self.q_size].reshape(t * self.total_num_heads, self.head_dim)
        ).reshape(t, self.q_size)
        qkv[:, self.q_size:self.q_size + self.kv_size] = self.k_norm(
            qkv[:, self.q_size:self.q_size + self.kv_size]
            .reshape(t * self.num_kv_heads, self.head_dim)
        ).reshape(t, self.kv_size)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, model_config: ModelConfig,
                 layer_idx: int, kv_cache_config: Optional[Dict] = None) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, model_config, layer_idx, kv_cache_config)
        self.mlp = LlamaMLP(config, model_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_quant=False)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_quant=False)

    def forward(self, hidden_states: torch.Tensor,
                input_metadata: InputMetadata) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, input_metadata)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3Model(LlamaModel):
    """LlamaModel but built with Qwen3DecoderLayer; bypasses LlamaModel.__init__
    to avoid double-allocating Llama attention modules."""

    def __init__(self, config: Qwen3Config, model_config: ModelConfig,
                 quant_kv_cache: bool = True,
                 kv_cache_config: Optional[Dict] = None) -> None:
        nn.Module.__init__(self)  # skip LlamaModel.__init__
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.model_config = model_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, model_config, i, kv_cache_config)
            if quant_kv_cache else None
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    # forward() is inherited unchanged from LlamaModel


class Qwen3ForCausalLM(LlamaForCausalLM):
    """Qwen3 causal LM; reuses LlamaForCausalLM.{forward,sample,load_weights}."""

    def __init__(
        self,
        config: Qwen3Config,
        model_config: ModelConfig,
        sampling_params: SamplingParams,
        quant_config: Optional[QServeQuantConfig] = QServeQuantConfig(weight_bits=4),
        kv_cache_config: Optional[Dict] = None,
        quant_path: Optional[str] = None,
    ) -> None:
        nn.Module.__init__(self)  # skip LlamaForCausalLM.__init__
        self.config = config
        self.quant_config = quant_config
        self.model_config = model_config
        self.model = Qwen3Model(config, model_config, kv_cache_config=kv_cache_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._column_parallel_layers: List[str] = []
        self._row_parallel_layers: List[str] = ["o_proj", "down_proj"]
        self.sampler = Sampler(sampling_params)

        tp_size = 1
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, 'head_dim',
                                config.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if quant_path is not None:
            print(f"[Qwen3] Loading weights from: {quant_path}")
            self.load_weights(quant_path)
            print(f"[Qwen3] Weights loaded successfully.")
    # load_weights / forward / sample inherited from LlamaForCausalLM
