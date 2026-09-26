#!/bin/bash

set -e

# =========================
# Model configuration
# =========================
MODEL="/NVME1/wenv/Qwen3-14B"
SERVED_MODEL_NAME="Qwen3-14B"

GPU="2,3"
PORT=8001
TP=2

MAX_MODEL_LEN=20480
MEM_FRACTION_STATIC=0.85
ATTENTION_BACKEND="flashinfer"

# =========================
# Decode CUDA graph configuration
# =========================
# Match the runtime NONSD buckets used by the toggleable engine so the pure
# NONSD baseline covers the same 1/2/4/8/16/32/64/128 shapes.
CUDA_GRAPH_BS_DECODE=(1 2 4 8 16 32 64 128)

# =========================
# Runtime configuration
# =========================
CONDA_ENV="sglang-0514"

LOG_FILE="/home/user/sglang-v0.5.14/logs/serve_sglang_nonsd.log"
PID_FILE="/home/user/sglang-v0.5.14/logs/serve_sglang_nonsd.pid"

CACHE_ROOT="${CACHE_ROOT:-/home/user/.cache/sglang-nonsd-engine}"

# =========================
# Validate model paths
# =========================
if [ ! -d "$MODEL" ]; then
    echo "❌ Target model directory does not exist:"
    echo "   $MODEL"
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$CACHE_ROOT"

# =========================
# Launch info
# =========================
echo "Launching SGLang engine:"
echo "  Target model:           $MODEL"
echo "  Served model name:      $SERVED_MODEL_NAME"
echo "  Port:                   $PORT"
echo "  GPU:                    $GPU"
echo "  Tensor parallel size:   $TP"
echo "  Max context length:     $MAX_MODEL_LEN"
echo "  Memory fraction:        $MEM_FRACTION_STATIC"
echo "  Decode CUDA graph bs:   ${CUDA_GRAPH_BS_DECODE[*]}"
echo "  Log file:               $LOG_FILE"

# =========================
# Activate conda environment
# =========================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$CACHE_ROOT"

# =========================
# Launch server (no speculative decoding)
# =========================
nohup python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp-size "$TP" \
    --context-length "$MAX_MODEL_LEN" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --attention-backend "$ATTENTION_BACKEND" \
    --download-dir "$CACHE_ROOT" \
    --cuda-graph-bs-decode "${CUDA_GRAPH_BS_DECODE[@]}" \
    --skip-server-warmup \
    > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo ""
echo "✅ SGLang engine launched."
echo "   PID: $PID"
echo ""
echo "💡 View logs:"
echo "   tail -f $LOG_FILE"
echo ""
echo "💡 Check process:"
echo "   ps -fp $PID"
echo ""
echo "💡 Kill process:"
echo "   kill $PID"
