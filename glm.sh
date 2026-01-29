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

# vLLM optimized settings for GLM-4.7
# - -O3: torch.compile optimization level 3 (best performance)
# - FlashInfer backend for optimized attention
# - MTP speculative decoding for faster generation
# - Prefix caching for repeated prompts
# - KV cache auto dtype for memory efficiency
vllm serve zai-org/GLM-4.7-FP8 \
    -O3 \
    --tensor-parallel-size 4 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 1 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.7-fp8 \
    --max-num-seqs 64 \
    --seed 42 \
    --max-model-len $CONTEXT_LENGTH \
    --host 0.0.0.0 \
    --port 8886 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 65536 \
    --enable-prefix-caching \
    --attention-backend FLASHINFER \
    --kv-cache-dtype auto \
    --enable-chunked-prefill \
    --disable-log-requests
