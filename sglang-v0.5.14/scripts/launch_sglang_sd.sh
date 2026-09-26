#!/bin/bash

set -e

# =========================
# Model configuration
# =========================
MODEL="/NVME1/wenv/Qwen3-14B"
DRAFT_MODEL="/NVME1/wenv/Qwen3-14B-eagle3-new"
SERVED_MODEL_NAME="Qwen3-14B"
GPU="0,1"
PORT=8001
TP=2

MAX_MODEL_LEN=20480
MEM_FRACTION_STATIC=0.85
ATTENTION_BACKEND="flashinfer"

# =========================
# Initial SD configuration
# =========================
SPECULATIVE_ALGORITHM="EAGLE3"

# SD 模式使用 3 个 draft steps；运行时可通过控制接口切到真正的 NONSD。
SPECULATIVE_NUM_STEPS=3

# Adaptive EAGLE 当前要求 topk=1
SPECULATIVE_EAGLE_TOPK=1

# EAGLE3 会根据 steps/topk 校正实际 draft token 数。
SPECULATIVE_NUM_DRAFT_TOKENS=1

ADAPTIVE_CONFIG="/home/user/sglang-v0.5.14/sglang_sd_settings.json"

# =========================
# Runtime configuration
# =========================
CONDA_ENV="sglang-0514"

LOG_FILE="/home/user/sglang-v0.5.14/logs/serve_sglang_sd.log"
PID_FILE="/home/user/sglang-v0.5.14/logs/serve_sglang_sd.pid"
CACHE_ROOT="${CACHE_ROOT:-/home/user/.cache/sglang-engine}"

# =========================
# Validate model paths
# =========================
if [ ! -d "$MODEL" ]; then
    echo "❌ Target model directory does not exist:"
    echo "   $MODEL"
    exit 1
fi

if [ ! -d "$DRAFT_MODEL" ]; then
    echo "❌ Draft model directory does not exist:"
    echo "   $DRAFT_MODEL"
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$CACHE_ROOT"

echo "Launching SGLang runtime SD/NONSD engine:"
echo "  Target model:           $MODEL"
echo "  Draft model:            $DRAFT_MODEL"
echo "  Served model name:      $SERVED_MODEL_NAME"
echo "  Port:                   $PORT"
echo "  GPU:                    $GPU"
echo "  Tensor parallel size:   $TP"
echo "  Max context length:     $MAX_MODEL_LEN"
echo "  Memory fraction:        $MEM_FRACTION_STATIC"
echo "  Speculative algorithm:  $SPECULATIVE_ALGORITHM"
echo "  Speculative steps:      $SPECULATIVE_NUM_STEPS"
echo "  Adaptive candidates:    [3]"
echo "  Skip draft extend:      1"
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
# Launch server
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
    --speculative-algorithm "$SPECULATIVE_ALGORITHM" \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-num-steps "$SPECULATIVE_NUM_STEPS" \
    --speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK" \
    --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS" \
    --skip-server-warmup \
    --enable-metrics \
    --max-running-requests 256 \
    --cuda-graph-bs-decode \
    1 2 4 8 16 32 40 48 56 64 \
    80 96 112 128 \
    > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo ""
echo "✅ SGLang runtime SD/NONSD engine launched."
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
