# model_name="Llama-3-8B-Instruct-Gradient-1048k"
# model_name="Qwen/Qwen3-8B"
attn_path=./attn_patterns/Qwen3-8B
model_path=Qwen/Qwen3-8B

#attn_path=./attn_patterns/Llama-3-8B-Instruct-Gradient-1048k
#model_path=./models/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor


#if [ ! -d "$model_path" ]; then
#  mkdir -p models
#  huggingface-cli download \
#    mit-han-lab/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor \
#    --local-dir $model_path \
#    --local-dir-use-symlinks False
#fi

# For w16a16, weights are loaded directly from HuggingFace; no local download needed.

precision="w16a16kv8"

NUM_RETRIEVAL_GPU_PAGE_BLOCKS=3000 \
NUM_STREAMING_GPU_PAGE_BLOCKS=200 \
python lserve_e2e_generation.py \
  --model $model_path \
  --ifb-mode \
  --precision $precision \
  --group-size -1 \
  --max-num-batched-tokens 4195000 \
  --max-num-seqs 1 \
  --omit-prompt \
  --kv-quant-granularity "per_tensor" \
  --chunk-prefill-size 32000 \
  --multiblock-switch 2048 \
  --static-sparse-attn-load-dir $attn_path \
  --static-sparsity 0.5 \
  --sparse-context-mode \
  --sparse-decode-mode 1 \
  --ctx-sink-token 128 \
  --ctx-local-token 8192 \
  --dec-sink-token 128 \
  --dec-local-token 256 \
  --sub-chunk-per-block 4 \
  --dynamic-sparse-token-budget 4096 \
  --selector-update-interval 4
