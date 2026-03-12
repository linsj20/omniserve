import torch
from einops import repeat
from typing import Optional, Union
from omniserve.attn_config import SpAttnConfig
from omniserve.modeling.models.llama_w4a8_unpad import LlamaForCausalLM as LlamaForCausalLMW4A8
from omniserve.modeling.models.llama_w8a8_unpad import LlamaForCausalLM as LlamaForCausalLMW8A8
from omniserve.modeling.models.llama_w16a16_unpad import LlamaForCausalLM as LlamaForCausalLMW16A16
from omniserve.modeling.models.mixtral_w4a8_unpad import MixtralForCausalLM as MixtralForCausalLMW4A8
from omniserve.modeling.models.qwen3_w16a16_unpad import Qwen3ForCausalLM as Qwen3ForCausalLMW16A16


def init_ctx_sparse_attn(
    model: Union[LlamaForCausalLMW4A8, LlamaForCausalLMW8A8, LlamaForCausalLMW16A16, MixtralForCausalLMW4A8, Qwen3ForCausalLMW16A16],
    sp_attn_config: SpAttnConfig,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    sparse_kv_cache_enabled = sp_attn_config.sparse_kv_cache_enabled()
    sparse_context_enabled = sp_attn_config.sparse_context_enabled()
    full_attention_heads = sp_attn_config.get_full_attention_heads()
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = torch.tensor(full_attention_heads[idx], device=device, dtype=dtype)
        if not sparse_kv_cache_enabled or not sparse_context_enabled or torch.all(layer_full_attention_heads == 1):
            head_mask_type = None
            streaming_info = None
        else:
            # print("[enable context sparse attn]")
            num_heads = model.total_num_heads
            num_kv_heads = model.total_num_kv_heads
            kv_repeat = num_heads // num_kv_heads
            layer_full_attention_heads = repeat(layer_full_attention_heads, 'h -> (h r)', r=kv_repeat)

            head_mask_type = torch.where(
                layer_full_attention_heads == 0, 
                torch.tensor(-1, dtype=torch.int32), 
                torch.tensor(0, dtype=torch.int32)
            ).to(device)

            context_sink_token = sp_attn_config.get_ctx_sink_size()
            context_local_token = sp_attn_config.get_ctx_local_size()

            streaming_info = torch.tensor(
                [context_sink_token, context_local_token]
                * num_heads, 
                device=device, 
                dtype=torch.int32
            )
            # print(f"streaming_info: {streaming_info}")
        module.register_buffer("head_mask_type", head_mask_type)
        module.register_buffer("streaming_info", streaming_info)

            

def init_sparse_kv_cache(
    model: Union[LlamaForCausalLMW4A8, LlamaForCausalLMW8A8, LlamaForCausalLMW16A16, MixtralForCausalLMW4A8, Qwen3ForCausalLMW16A16],
    sp_attn_config: SpAttnConfig,
):
    def transform_sequence(tensor):
        zeros_mask = (tensor == 0)
        ones_mask = (tensor == 1)
        result = torch.empty_like(tensor)
        result[zeros_mask] = torch.arange(zeros_mask.sum(), device=tensor.device, dtype=torch.int32)
        result[ones_mask] = torch.arange(ones_mask.sum(), device=tensor.device, dtype=torch.int32)
        return result
    
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    full_attention_heads = sp_attn_config.get_full_attention_heads()
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = torch.tensor(full_attention_heads[idx], device=device, dtype=torch.int32)
        head_rank_table = transform_sequence(layer_full_attention_heads)
        # print(f"layer_full_attention_heads: {layer_full_attention_heads}")
        pooling_heads_idx = torch.where(layer_full_attention_heads == 1)[0].to(torch.int32).to(device)

        module.register_buffer("retrieval_head_flags", layer_full_attention_heads)
        module.register_buffer("head_rank_table", head_rank_table)
        module.register_buffer("pooling_heads_idx", pooling_heads_idx)

        module.num_retrieval_kv_heads = sp_attn_config.retrieval_head_num(idx)
        module.num_streaming_kv_heads = sp_attn_config.streaming_head_num(idx)
        module.sink_blocks = sp_attn_config.get_dec_sink_block_num()
        module.local_blocks = sp_attn_config.get_dec_local_block_num()
        module.sink_size = sp_attn_config.get_dec_sink_size()
        module.local_size = sp_attn_config.get_dec_local_size()
              