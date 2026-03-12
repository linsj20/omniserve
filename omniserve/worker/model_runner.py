# original file: https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py
# modified by: Haotian Tang, Shang Yang, Zhekai Zhang
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
# @article{yang2025lserve,
#   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
#   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
#   year={2025}
# }

import os
from typing import Dict, List, Optional, Tuple, Union

import omniserve_backend.fused_attention_fine_grained_dense as fused_attention
import torch

from omniserve.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)
from omniserve.logger import init_logger
from omniserve.modeling.models.llama_w4a8_unpad import (
    LlamaForCausalLM as LlamaForCausalLMW4A8,
)
from omniserve.modeling.models.llama_w8a8_unpad import (
    LlamaForCausalLM as LlamaForCausalLMW8A8,
)
from omniserve.modeling.models.llama_w16a16_unpad import (
    LlamaForCausalLM as LlamaForCausalLMW16A16,
)
from omniserve.modeling.models.mixtral_w4a8_unpad import (
    MixtralForCausalLM as MixtralForCausalLMW4A8,
)
from omniserve.modeling.models.qwen3_w16a16_unpad import (
    Qwen3ForCausalLM as Qwen3ForCausalLMW16A16,
)
from omniserve.sampling_params import SamplingParams
from omniserve.sequence import SamplerOutput, SequenceGroupMetadata
from omniserve.utils.input_metadata import InputMetadata
from omniserve.utils.utils import STR_DTYPE_TO_TORCH_DTYPE
from omniserve.worker.cache_engine import CacheEngine

from omniserve.modeling.layers.ctx_attn.ctx_attn_init import init_ctx_sparse_attn, init_sparse_kv_cache
from omniserve.modeling.layers.ctx_attn.block_table_utils import pad_block_tables, get_layer_block_tables, _make_tensor_with_pad

logger = init_logger(__name__)


# def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
#     assert len(x) <= max_len
#     return x + [pad] * (max_len - len(x))


# def _make_tensor_with_pad(
#     x: List[List[int]],
#     max_len: int,
#     pad: int,
#     dtype: torch.dtype,
#     device: Optional[Union[str, torch.device]],
# ) -> torch.Tensor:
#     padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
#     return torch.tensor(padded_x, dtype=dtype, device=device)


class ModelRunner:
    _sizeof = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int8: 1}

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        kv_cache_dtype: torch.dtype,
        is_driver_worker: bool = False,
        precision: str = "w4a8kv4",
        kv_cache_config: Optional[Dict] = None,
        quant_path: Optional[str] = None,
        group_size: int = -1,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.is_driver_worker = is_driver_worker
        self.kv_cache_config = kv_cache_config

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (
            model_config.get_sliding_window() if model_config is not None else None
        )
        self.device_config = (
            device_config if device_config is not None else DeviceConfig()
        )
        self.device = self.device_config.device
        # Note: Shang's important fix here. Otherwise non-GEMM part will run in FP32.
        model_type = model_config.hf_config.architectures[0]

        if model_type == "LlamaForCausalLM" or model_type == "MistralForCausalLM":
            if "w4a8" in precision:
                print(f"[INFO] Using {precision} precision")
                self.model = (
                    LlamaForCausalLMW4A8(
                        self.model_config.hf_config,
                        self.model_config,
                        group_size,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=quant_path,
                    )
                    .half()
                    .to(self.device)
                )
            elif "w8a8" in precision:
                print(f"[INFO] Using {precision} precision")
                self.model = (
                    LlamaForCausalLMW8A8(
                        self.model_config.hf_config,
                        self.model_config,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=quant_path,
                    )
                    .half()
                    .to(self.device)
                )
            elif "w16a16" in precision:
                print(f"[INFO] Using {precision} precision")
                # For w16a16 there is no quantized checkpoint; load from the HF
                # model ID (or local path) directly when no --quant-path given.
                effective_quant_path = quant_path if quant_path is not None else self.model_config.model
                self.model = (
                    LlamaForCausalLMW16A16(
                        self.model_config.hf_config,
                        self.model_config,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=effective_quant_path,
                    )
                    .half()
                    .to(self.device)
                )
            else:
                raise ValueError(
                    f"Unsupported model precision: {precision}. Expected w8a8 or w4a8."
                )
        elif model_type == "MixtralForCausalLM":
            if "w4a8" in precision:
                print(f"[INFO] Using {precision} precision")
                self.model = (
                    MixtralForCausalLMW4A8(
                        self.model_config.hf_config,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=quant_path,
                    )
                    .half()
                    .to(self.device)
                )
            else:
                raise ValueError(
                    f"Unsupported model precision: {precision}. Expected w4a8."
                ) # add by JXGuo: secure the model to be CausalLM
        elif model_type == "Qwen3ForCausalLM":
            if "w16a16" in precision:
                print(f"[INFO] Using {precision} precision")
                # For w16a16 there is no quantized checkpoint; load from the HF
                # model ID (or local path) directly when no --quant-path given.
                effective_quant_path = quant_path if quant_path is not None else self.model_config.model
                self.model = (
                    Qwen3ForCausalLMW16A16(
                        self.model_config.hf_config,
                        self.model_config,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=effective_quant_path,
                    )
                    .half()
                    .to(self.device)
                )
            else:
                raise ValueError(
                    f"Unsupported model precision: {precision}. Expected w16a16."
                )
        else:
            raise ValueError(f"Unsupported model type: {model_type}.")
        self.block_size = None  # Set after initial profiling.

        init_ctx_sparse_attn(
            model=self.model,
            sp_attn_config=self.model_config.sp_attn_config,
        )
        init_sparse_kv_cache(
            model=self.model,
            sp_attn_config=self.model_config.sp_attn_config,
        )
        
        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None
            else 0
        )

        self.kv_cache_dtype = kv_cache_dtype
        self.num_layers = model_config.get_num_layers(parallel_config)

        kv_scale_layer_offsets = (
            torch.arange(
                self.model_config.get_num_layers(self.parallel_config)
            ).unsqueeze(0)
            * self.model_config.max_model_len
            * self.model_config.get_num_kv_heads(self.parallel_config)
        )
        kv_scale_kv_offsets = (
            torch.arange(2).unsqueeze(1)
            * self.model_config.get_num_layers(self.parallel_config)
            * self.model_config.max_model_len
            * self.model_config.get_num_kv_heads(self.parallel_config)
        )
        self.kv_scale_offsets = (kv_scale_layer_offsets + kv_scale_kv_offsets).to(
            self.device_config.device
        )

        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # cache_block_mem_size = CacheEngine.get_cache_block_size(
        #     cache_config.block_size,
        #     cache_config.cache_bits,
        #     self.model_config,
        #     self.parallel_config,
        # )
        # num_gpu_blocks = int(
        #     (
        #         free_gpu_memory
        #         - total_gpu_memory * (1 - cache_config.gpu_memory_utilization)
        #     )
        #     // cache_block_mem_size
        # )
        num_retrieval_cpu_blocks = 10
        num_streaming_cpu_blocks = 10

        manual_num_retrieval_gpu_blocks = os.environ.get("NUM_RETRIEVAL_GPU_PAGE_BLOCKS")
        manual_num_streaming_gpu_blocks = os.environ.get("NUM_STREAMING_GPU_PAGE_BLOCKS")
        if manual_num_retrieval_gpu_blocks is not None:
            num_retrieval_gpu_blocks = int(manual_num_retrieval_gpu_blocks)
        if manual_num_streaming_gpu_blocks is not None:
            num_streaming_gpu_blocks = int(manual_num_streaming_gpu_blocks)

        cache_config.num_retrieval_gpu_blocks = num_retrieval_gpu_blocks
        cache_config.num_streaming_gpu_blocks = num_streaming_gpu_blocks
        cache_config.num_retrieval_cpu_blocks = num_retrieval_cpu_blocks
        cache_config.num_streaming_cpu_blocks = num_streaming_cpu_blocks
        logger.info(
            # f"# GPU blocks: {num_gpu_blocks}, " f"# CPU blocks: {num_cpu_blocks}"
            f"# Retrieval GPU blocks: {num_retrieval_gpu_blocks}, " f"# Streaming GPU blocks: {num_streaming_gpu_blocks}" #f"# Retrieval CPU blocks: {num_retrieval_cpu_blocks}, " f"# Streaming CPU blocks: {num_streaming_cpu_blocks}"
        )
        self.cache_engine = CacheEngine(
            cache_config, model_config, parallel_config, kv_cache_config
        )
        # self.cache_events = self.cache_engine.events
        # self.gpu_cache = self.cache_engine.gpu_cache
        self.cache_config = cache_config

    def load_model(self) -> None:
        vocab_size = self.model.config.vocab_size

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        max_num_blocks = (
            self.max_context_len_to_capture + block_size - 1
        ) // block_size

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        ifb_mode: bool = True,
    ) -> Tuple[torch.Tensor, InputMetadata,]:
        # print("[in _prepare_prompt]")
        # kentang-mit@: let's assume that prefix is always none
        assert len(seq_group_metadata_list) > 0
        input_tokens = []
        retrieval_context_lens = []
        streaming_context_lens = []
        retrieval_block_tables = []
        streaming_block_tables = []
        kv_scales_ptrs = []
        sink_size, local_size = self.model_config.sp_attn_config.get_dec_sink_size(), self.model_config.sp_attn_config.get_dec_sink_size()
        sink_block, local_block = self.model_config.sp_attn_config.get_dec_sink_block_num(), self.model_config.sp_attn_config.get_dec_local_block_num()
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            input_tokens.append(prompt_tokens)
            context_len = len(prompt_tokens)
            retrieval_context_lens.append(context_len)

            if seq_group_metadata.retrieval_block_tables is not None:
                retrieval_block_table = seq_group_metadata.retrieval_block_tables[seq_id]
                retrieval_block_tables.append(retrieval_block_table)
                
            if self.model_config.sp_attn_config.sparse_kv_cache_enabled():
                streaming_context_lens.append(min(context_len, sink_size + local_size))
                if seq_group_metadata.streaming_block_tables is not None:
                    streaming_block_table = seq_group_metadata.streaming_block_tables[seq_id]
                    if context_len > sink_size + local_size:   
                        streaming_block_table = streaming_block_table[:sink_block] + streaming_block_table[-local_block:]
                    streaming_block_tables.append(streaming_block_table)

        max_prompt_len = max(retrieval_context_lens)
        input_tokens = torch.cat([torch.tensor(x) for x in input_tokens], dim=0).to(
            device=self.device
        )
        retrieval_context_lens_tensor = torch.tensor(
            retrieval_context_lens, dtype=torch.int, device=self.device
        )
        
        if self.model_config.sp_attn_config.sparse_kv_cache_enabled():
            streaming_context_lens_tensor = torch.tensor(
                streaming_context_lens, dtype=torch.int, device=self.device
            )
        else:
            streaming_context_lens_tensor = None
        
        cu_seqlens_tensor = torch.cumsum(retrieval_context_lens_tensor, dim=0).int()
        cu_seqlens_tensor = torch.nn.functional.pad(cu_seqlens_tensor, (1, 0), value=0)
        # Prepare prefix block tables
        (
            retrieval_block_tables, 
            streaming_block_tables, 
            max_retrieval_block_table_len, 
            max_streaming_block_table_len
        ) = pad_block_tables(
            retrieval_block_tables=retrieval_block_tables,                
            streaming_block_tables=streaming_block_tables, 
            sparse_kv_cache_enabled=self.model_config.sp_attn_config.sparse_kv_cache_enabled(),
            device="cpu",
        )
        
        layers = self.num_layers
        (
            layer_retrieval_block_tables, 
            layer_streaming_block_tables
        ) = get_layer_block_tables(
            cache_engine=self.cache_engine,
            layers=layers,
            cache_config=self.cache_config,
            retrieval_block_tables=retrieval_block_tables,
            streaming_block_tables=streaming_block_tables,
            sparse_kv_cache_enabled=self.model_config.sp_attn_config.sparse_kv_cache_enabled(),
            device=self.device
        )
        

        padding_offsets_tensor = fused_attention.compute_padding_offsets(
            cu_seqlens_tensor, max_prompt_len, input_tokens.size(0)
        )

        input_metadata = InputMetadata(
            is_prompt=True,
            retrieval_context_lens=retrieval_context_lens_tensor,
            streaming_context_lens=streaming_context_lens_tensor,
            padding_offsets=padding_offsets_tensor,
            cu_seqlens=cu_seqlens_tensor,
            max_seq_len=max_prompt_len,
            max_retrieval_block_table_len=max_retrieval_block_table_len,
            max_streaming_block_table_len=max_streaming_block_table_len,
            retrieval_block_tables=layer_retrieval_block_tables,
            streaming_block_tables=layer_streaming_block_tables,
            kv_cache_dtype=self.kv_cache_dtype,
            kv_scales=None,
            batched_seq_len=input_tokens.size(0),
            model=self.model,
        )
        return (input_tokens, input_metadata)

    def _prepare_decode_ifb(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, InputMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens = []
        retrieval_context_lens = []
        streaming_context_lens = []
        retrieval_block_tables = []
        streaming_block_tables = []
        assert self.sliding_window is None
        sink_size, local_size = self.model_config.sp_attn_config.get_dec_sink_size(), self.model_config.sp_attn_config.get_dec_sink_size()
        sink_block, local_block = self.model_config.sp_attn_config.get_dec_sink_block_num(), self.model_config.sp_attn_config.get_dec_local_block_num()
        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])
                context_len = seq_data.get_len()
                retrieval_context_lens.append(context_len)
                retrieval_block_table = seq_group_metadata.retrieval_block_tables[seq_id]
                retrieval_block_tables.append(retrieval_block_table)
                
                if self.model_config.sp_attn_config.sparse_kv_cache_enabled():
                    streaming_context_lens.append(min(context_len, sink_size + local_size))
                    streaming_block_table = seq_group_metadata.streaming_block_tables[seq_id]
                    if context_len > sink_size + local_size: 
                        streaming_block_table = streaming_block_table[:sink_block] + streaming_block_table[-local_block:]
                    streaming_block_tables.append(streaming_block_table)

        max_context_len = max(retrieval_context_lens)

        input_tokens = _make_tensor_with_pad(
            input_tokens, max_len=1, pad=0, dtype=torch.long, device=self.device
        ).squeeze(1)
        retrieval_context_lens = torch.tensor(retrieval_context_lens, dtype=torch.int, device=self.device)
        streaming_context_lens = torch.tensor(streaming_context_lens, dtype=torch.int, device=self.device)
        
        (
            retrieval_block_tables, 
            streaming_block_tables, 
            max_retrieval_block_table_len, 
            max_streaming_block_table_len
        ) = pad_block_tables(
            retrieval_block_tables=retrieval_block_tables,                
            streaming_block_tables=streaming_block_tables, 
            sparse_kv_cache_enabled=self.model_config.sp_attn_config.sparse_kv_cache_enabled(),
            device=self.device,
        )

        layers = self.num_layers

        (
            layer_retrieval_block_tables, 
            layer_streaming_block_tables
        ) = get_layer_block_tables(
            cache_engine=self.cache_engine,
            layers=layers,
            cache_config=self.cache_config,
            retrieval_block_tables=retrieval_block_tables,
            streaming_block_tables=streaming_block_tables,
            sparse_kv_cache_enabled=self.model_config.sp_attn_config.sparse_kv_cache_enabled(),
            device=self.device
        )

        input_metadata = InputMetadata(
            is_prompt=False,
            cu_seqlens=None,
            padding_offsets=None,
            retrieval_context_lens=retrieval_context_lens,
            streaming_context_lens=streaming_context_lens,
            max_seq_len=max_context_len,
            max_retrieval_block_table_len=max_retrieval_block_table_len,
            max_streaming_block_table_len=max_streaming_block_table_len,
            retrieval_block_tables=layer_retrieval_block_tables,
            streaming_block_tables=layer_streaming_block_tables,
            kv_scales=None,
            kv_cache_dtype=self.kv_cache_dtype,
            batched_seq_len=input_tokens.size(0),
            model=self.model,
        )

        return (input_tokens, input_metadata)

    def _prepare_decode_no_ifb(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        layer_retrieval_block_tables: List[torch.Tensor],
        layer_streaming_block_tables: List[torch.Tensor],
        max_retrieval_block_table_len: int,
        max_streaming_block_table_len: int,
        layer_kv_scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int], List[int]]:
        # print("[in _prepare_decode_no_ifb]")
        # print(f"layer_retrieval_block_tables: {layer_retrieval_block_tables}")
        # print(f"layer_streaming_block_tables: {layer_streaming_block_tables}")
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        retrieval_context_lens: List[int] = []
        streaming_context_lens: List[int] = []
        assert not seq_group_metadata_list[0].is_prompt

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

        # NOTE: We assume that all sequences have the same length.
        seq_len = seq_group_metadata_list[0].seq_data[0].get_len()
        retrieval_context_len = seq_len
        retrieval_context_lens = torch.tensor(
            [retrieval_context_len,] * len(seq_group_metadata_list),
            dtype=torch.int,
            device=self.device,
        )
        max_context_len = retrieval_context_len
        
        if self.model_config.sp_attn_config.sparse_kv_cache_enabled():
            sink_size, local_size = self.model_config.sp_attn_config.get_dec_sink_size(), self.model_config.sp_attn_config.get_dec_sink_size()
            streaming_context_len = min(retrieval_context_len, sink_size + local_size)
            streaming_context_lens = torch.tensor(
                [streaming_context_len,] * len(seq_group_metadata_list),
                dtype=torch.int,
                device=self.device,
            )
        else:
            streaming_context_lens = None
        

        input_tokens = _make_tensor_with_pad(
            input_tokens, max_len=1, pad=0, dtype=torch.long, device=self.device
        ).squeeze(1)
        input_metadata = InputMetadata(
            is_prompt=False,
            cu_seqlens=None,
            padding_offsets=None,
            retrieval_context_lens=retrieval_context_lens,
            streaming_context_lens=streaming_context_lens,
            max_seq_len=max_context_len,
            max_retrieval_block_table_len=max_retrieval_block_table_len,
            max_streaming_block_table_len=max_streaming_block_table_len,
            retrieval_block_tables=layer_retrieval_block_tables,
            streaming_block_tables=layer_streaming_block_tables,
            kv_scales=layer_kv_scales,
            kv_cache_dtype=self.kv_cache_dtype,
            batched_seq_len=input_tokens.size(0),
            model=self.model,
        )

        return (input_tokens, input_metadata)

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        ifb_mode: bool = True,
        layer_retrieval_block_tables: List[torch.Tensor] = None,
        layer_streaming_block_tables: List[torch.Tensor] = None,
        max_retrieval_block_table_len: int = None,
        max_streaming_block_table_len: int = None,
        layer_kv_scales: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, InputMetadata]:
        # NOTE: We assume that all sequences in the group are all prompts or all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        
        # NOTE: We assume that all sequences in the group share the same sampling_params.
        sampling_params = seq_group_metadata_list[0].sampling_params

        if len(sampling_params.decoding_sim_token_ids) > 0:
            assert len(seq_group_metadata_list) == 1, "Only one sequence per-batch is allowed when activating decoding simulation."

        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_metadata) = self._prepare_prompt(
                seq_group_metadata_list, ifb_mode
            )
        else:
            if ifb_mode:
                (input_tokens, input_metadata) = self._prepare_decode_ifb(
                    seq_group_metadata_list
                )
            else:
                (input_tokens, input_metadata) = self._prepare_decode_no_ifb(
                    seq_group_metadata_list,
                    layer_retrieval_block_tables,
                    layer_streaming_block_tables,
                    max_retrieval_block_table_len,
                    max_streaming_block_table_len,
                    layer_kv_scales=layer_kv_scales,
                )

        return (input_tokens, input_metadata, sampling_params)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        ifb_mode: bool = True,
        layer_retrieval_block_tables: List[torch.Tensor] = None,
        layer_streaming_block_tables: List[torch.Tensor] = None,
        max_retrieval_block_table_len: int = None,
        max_streaming_block_table_len: int = None,
        layer_kv_scales: torch.Tensor = None,
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_metadata, sampling_params) = self.prepare_input_tensors(
            seq_group_metadata_list,
            ifb_mode,
            layer_retrieval_block_tables,
            layer_streaming_block_tables,
            max_retrieval_block_table_len,
            max_streaming_block_table_len,
            layer_kv_scales=layer_kv_scales,
        )
        model = self.model
        # return None
        output = model(input_tokens, input_metadata)
        tokens = model.sample(input_tokens, output, input_metadata, sampling_params)
        return tokens
