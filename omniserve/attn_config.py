from typing import Optional, Union
import os
import torch
import numpy as np
import json
from einops import repeat


class CtxAttnConfig:
    """Context Sparse Attention configuration.
    
    Args:
        
    """
    def __init__(
        self,
        sparse_context_mode: bool,
        ctx_sink_token: Optional[int] = 0,
        ctx_local_token: Optional[int] = 0,
    ) -> None:
        self.sparse_context_mode = sparse_context_mode
        self.ctx_sink_token = ctx_sink_token
        self.ctx_local_token = ctx_local_token
        
        self._verify_args()
    
    def _verify_args(self) -> None:
        if self.sparse_context_mode:
            if self.ctx_sink_token < 0 or self.ctx_local_token < 0:
                raise ValueError(f"Sink and local token size must be non-negative. Got {self.ctx_sink_token} and {self.ctx_local_token}.")
        


class DecAttnConfig:
    """Sparse Attention configuration.
    
    Args:
        static_sparse_attn_load_dir: Directory to load the attention pattern.
        static_sparsity: static_sparsity of the attention pattern. (how many percent is streaming heads)
        todo:
        assignable sink/local size for each head?
        switch point for streaming+dense to streaming+blocksparse?
        
    """
    def __init__(
        self,
        sparse_decode_mode: int,
        cache_block_size: int,
        dec_sink_token: Optional[int] = 0,
        dec_local_token: Optional[int] = 0,
        sub_chunk_per_block: Optional[int] = 0,
        dynamic_sparse_token_budget: Optional[int] = 0,
        selector_update_interval: Optional[int] = 0
    ) -> None:
        self.sparse_decode_mode = sparse_decode_mode
        self.cache_block_size = cache_block_size
        self.dec_sink_token = dec_sink_token
        self.dec_local_token = dec_local_token
        
        self.sub_chunk_per_block = sub_chunk_per_block
        self.dynamic_sparse_token_budget = dynamic_sparse_token_budget
        self.selector_update_interval = selector_update_interval
        self.dec_sink_block = self.dec_sink_token // self.cache_block_size
        self.dec_local_block = self.dec_local_token // self.cache_block_size + 1

        self._verify_args()
             
    def _verify_args(self) -> None:
        if not self.sparse_decode_mode == 0:
            if self.dec_sink_token % self.cache_block_size != 0:
                raise ValueError(f"Sink token size must be divisible by cache block size. Got {self.dec_sink_token} and {self.cache_block_size}.")
            if self.dec_local_token % self.cache_block_size != 0:
                raise ValueError(f"Local token size must be divisible by cache block size. Got {self.dec_local_token} and {self.cache_block_size}.")
            if not self.sub_chunk_per_block > 0:
                raise ValueError(f"Sub chunk per block must be larger than 0. Got {self.sub_chunk_per_block}.")
            if self.dynamic_sparse_token_budget <= 0 or self.dynamic_sparse_token_budget % self.cache_block_size != 0:
                raise ValueError(f"require self.dynamic_sparse_token_budget > 0 and self.dynamic_sparse_token_budget % self.cache_block_size == 0. Got {self.dynamic_sparse_token_budget} and {self.cache_block_size}.")


class SpAttnConfig:
    def __init__(
        self,
        total_num_kv_heads,
        total_num_layers,
        cache_block_size,
        ctx_attn_config: CtxAttnConfig,
        dec_attn_config: DecAttnConfig,
        static_sparse_attn_load_dir: Optional[str] = None,
        static_sparsity: Optional[float] = 0.0,
    ):
        self.total_num_kv_heads = total_num_kv_heads
        self.total_num_layers = total_num_layers
        self.cache_block_size = cache_block_size
        self.ctx_attn_config = ctx_attn_config
        self.dec_attn_config = dec_attn_config
        self.static_sparse_attn_load_dir = static_sparse_attn_load_dir
        self.static_sparsity = static_sparsity
        
        self._verify_args()
        self._prepare_attn_pattern()
    
    def _verify_args(self) -> None:
        if self.static_sparse_attn_load_dir is None:
            if self.static_sparsity != 0:
                raise ValueError(f"Static sparsity is not 0 but no static sparse attention pattern file is provided.")
        else:
            if not os.path.exists(self.static_sparse_attn_load_dir):
                raise FileExistsError(f"static sparse attention pattern file: {self.static_sparse_attn_load_dir} does not exist.")
            if self.static_sparsity < 0 or self.static_sparsity > 1:
                raise ValueError(f"Static sparsity must be between 0 and 1. Got {self.static_sparsity}.")
        
    def _prepare_attn_pattern(self):
        def _sparsify_attention_heads(
            full_attention_heads, 
            static_sparsity,
        ):
            # add a very small random noise to full_attention_heads to break ties
            full_attention_heads += np.random.uniform(0, 1e-6, full_attention_heads.shape)

            if static_sparsity is not None:
                # ignore the threshold and use the static_sparsity
                # set the static_sparsity small values to 0 and others to 1
                threshold = np.quantile(full_attention_heads, static_sparsity)
            else:
                assert threshold is not None, "Either threshold or static_sparsity must be provided"

            if static_sparsity >= 1:
                # all heads are pruned
                threshold = 2
            if static_sparsity <= 0:
                # no heads are pruned
                threshold = -1

            full_attention_heads = (full_attention_heads >= threshold).astype(float)
            actual_sparsity = 1 - np.mean(full_attention_heads)
            print(f"actual sparsity: {actual_sparsity}")
            return full_attention_heads

    
        if self.static_sparse_attn_load_dir is None or self.static_sparsity == 0:
            self.static_sparsity = 0
            self.full_attention_heads = torch.ones(self.total_num_layers, self.total_num_kv_heads, dtype=torch.int32)
        else:
            full_attention_heads = np.loadtxt(
                os.path.join(self.static_sparse_attn_load_dir, "full_attention_heads.tsv"),
                dtype=float,
                delimiter="\t",
            )
            full_attention_heads = np.clip(full_attention_heads, 0, 1)
            self.full_attention_heads = torch.tensor(_sparsify_attention_heads(full_attention_heads, self.static_sparsity), dtype=torch.int32)
            config_path = os.path.join(self.static_sparse_attn_load_dir, "config.json")
            if os.path.exists(config_path):
                config = json.load(open(config_path))

    
    def retrieval_head_num(self, layer_idx):
        return self.full_attention_heads[layer_idx].sum().item()     
    
    def streaming_head_num(self, layer_idx):
        return self.total_num_kv_heads - self.retrieval_head_num(layer_idx) 
    
    def get_static_sparsity(self) -> float:
        return self.static_sparsity
    
    def get_full_attention_heads(self):
        return self.full_attention_heads
    
    def sparse_kv_cache_enabled(self) -> bool:
        return self.static_sparsity != 0
    
    def sparse_context_enabled(self) -> bool:
        return self.ctx_attn_config.sparse_context_mode
    
    def sparse_decode_enabled(self) -> bool:
        return self.dec_attn_config.sparse_decode_mode != 0
    
    def get_ctx_sink_size(self) -> int:
        return self.ctx_attn_config.ctx_sink_token
    
    def get_ctx_local_size(self) -> int:
        return self.ctx_attn_config.ctx_local_token
    
    def get_sparse_decode_mode(self) -> bool:
        return self.dec_attn_config.sparse_decode_mode
    
    def get_dec_sub_chunk_per_block(self) -> int:
        return self.dec_attn_config.sub_chunk_per_block
    
    def get_dec_dynamic_sparse_token_budget(self) -> int:
        return self.dec_attn_config.dynamic_sparse_token_budget
    
    def get_dec_selector_update_interval(self) -> int:
        return self.dec_attn_config.selector_update_interval
    
    def get_dec_sink_size(self) -> int:
        return self.dec_attn_config.dec_sink_token
    
    def get_dec_local_size(self) -> int:
        return self.dec_attn_config.dec_local_token
    
    def get_dec_sink_block_num(self) -> int:
        return self.dec_attn_config.dec_sink_block
    
    def get_dec_local_block_num(self) -> int:
        return self.dec_attn_config.dec_local_block
    
    
def sparse_attn_init(
    total_num_kv_heads: int,
    total_num_layers: int,
    cache_block_size: int,
    sparse_context_mode: bool,
    sparse_decode_mode: int,
    static_sparse_attn_load_dir: Optional[str] = None,
    static_sparsity: Optional[float] = 0.0,
    ctx_sink_token: Optional[int] = 0,
    ctx_local_token: Optional[int] = 0,
    dec_sink_token: Optional[int] = 0,
    dec_local_token: Optional[int] = 0,
    sub_chunk_per_block: Optional[int] = 0,
    dynamic_sparse_token_budget: Optional[int] = 0,
    selector_update_interval: Optional[int] = 0
) -> SpAttnConfig:
    ctx_attn_config = CtxAttnConfig(
        sparse_context_mode = sparse_context_mode, 
        ctx_sink_token = ctx_sink_token, 
        ctx_local_token = ctx_local_token
    )
    dec_attn_config = DecAttnConfig(
        sparse_decode_mode = sparse_decode_mode,
        cache_block_size = cache_block_size,
        dec_sink_token = dec_sink_token,
        dec_local_token = dec_local_token,
        sub_chunk_per_block = sub_chunk_per_block,
        dynamic_sparse_token_budget = dynamic_sparse_token_budget,
        selector_update_interval = selector_update_interval
    )
    sp_attn_config = SpAttnConfig(
        total_num_kv_heads = total_num_kv_heads,
        total_num_layers = total_num_layers,
        cache_block_size = cache_block_size,
        ctx_attn_config = ctx_attn_config,
        dec_attn_config = dec_attn_config,
        static_sparse_attn_load_dir = static_sparse_attn_load_dir,
        static_sparsity = static_sparsity
    )
    
    return sp_attn_config