#!/bin/bash
# Configuration
export HF_HOME="./workspace_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install uv
if [ ! -d ".venv" ]; then
    uv venv
fi
source .venv/bin/activate
uv pip install vllm flashinfer-python hf_transfer
CONTEXT_LENGTH=32768

# Auto-detect number of GPUs for tensor parallelism
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -gt 1 ]; then
    TP_SIZE=$NUM_GPUS
else
    TP_SIZE=1
fi

vllm serve tachytelicdetonation/medgemma-27b-it-fp8-static \
    --served-model-name tachytelicdetonation/medgemma-27b-it-fp8-static \
    --kv-cache-dtype "auto" \
    --max-num-seqs 64 \
    --seed 42 \
    --max-model-len $CONTEXT_LENGTH \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size $TP_SIZE \
    --disable-log-requests \
    --host 0.0.0.0 \
    --port 8886
