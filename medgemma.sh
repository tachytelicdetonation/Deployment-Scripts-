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

# vLLM v0.14.0+ optimized settings
# - V1 engine (default), FlashInfer backend
# - async-scheduling (default), prefix-caching
# - auto context length fits to available GPU memory
vllm serve tachytelicdetonation/medgemma-27b-it-fp8-static \
    -O3 \
    --served-model-name tachytelicdetonation/medgemma-27b-it-fp8-static \
    --max-model-len auto \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size $TP_SIZE \
    --kv-cache-dtype auto \
    --enable-prefix-caching \
    --attention-backend FLASHINFER \
    --disable-log-requests \
    --seed 42 \
    --host 0.0.0.0 \
    --port 8886
