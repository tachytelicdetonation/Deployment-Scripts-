#!/bin/bash
# Configuration
export HF_HOME="./workspace_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install uv
if [ ! -d ".venv" ]; then
    uv venv
fi
source .venv/bin/activate
uv pip install "vllm>=0.14.0" flashinfer-python hf_transfer

# Auto-detect number of GPUs for tensor parallelism
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -gt 1 ]; then
    TP_SIZE=$NUM_GPUS
else
    TP_SIZE=1
fi

CONTEXT_LENGTH=32768

# vLLM optimized settings for GLM-4.5-Air on H200
# - GLM-4.5-Air: 106B total params, 12B active (MoE)
# - H200 throughput optimizations for max tokens/s:
#   - FP8 KV cache doubles capacity for more batching
#   - Async scheduling eliminates GPU idle time
#   - Expert parallel for MoE efficiency
# - FlashInfer backend, prefix caching, thinking mode
vllm serve zai-org/GLM-4.5-Air-FP8 \
    -O3 \
    --tensor-parallel-size $TP_SIZE \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.5-air-fp8 \
    --max-num-seqs 256 \
    --seed 42 \
    --max-model-len $CONTEXT_LENGTH \
    --host 0.0.0.0 \
    --port 8887 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 65536 \
    --enable-prefix-caching \
    --attention-backend FLASHINFER \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-expert-parallel \
    --disable-log-requests
