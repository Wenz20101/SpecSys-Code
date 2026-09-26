#!/bin/bash

set -e

# =========================
# Model configuration
# =========================
MODEL="/NVME1/wenv/Qwen3-32B"
DRAFT_MODEL="/NVME1/wenv/Qwen3-32B-eagle3-sglang"
SERVED_MODEL_NAME="Qwen3-32B"

GPU="0,1,2,3"
PORT=8001
TP=4

MAX_MODEL_LEN=20480
MEM_FRACTION_STATIC=0.85
# ATTENTION_BACKEND="flashinfer"
ATTENTION_BACKEND="flashinfer"

# =========================
# Initial SD configuration
# =========================
SPECULATIVE_ALGORITHM="EAGLE3"

# SD 模式使用 5 个 draft steps；运行时可通过控制接口切到真正的 NONSD。
SPECULATIVE_NUM_STEPS=0

# Adaptive EAGLE 当前要求 topk=1
SPECULATIVE_EAGLE_TOPK=1

# EAGLE3 会根据 steps/topk 校正实际 draft token 数。
SPECULATIVE_NUM_DRAFT_TOKENS=1

ADAPTIVE_CONFIG="/home/user/sglang-v0.5.14/sglang_asd_settings.json"

# =========================
# Runtime configuration
# =========================
CONDA_ENV="sglang-0514"

LOG_FILE="/home/user/sglang-v0.5.14/logs/serve_sglang.log"
PID_FILE="/home/user/sglang-v0.5.14/logs/serve_sglang.pid"

CACHE_ROOT="${CACHE_ROOT:-/home/user/.cache/sglang-asd-engine}"

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

# =========================
# Keep ASD at step=5; SD/NONSD is controlled by the runtime endpoint.
# =========================
# 只设置一个 batch-size 区间，并且候选值只有 5，
# 因此 SD 模式始终使用 step=5。
cat > "$ADAPTIVE_CONFIG" <<'JSON'
{
  "ema_alpha": 0.2,
  "warmup_batches": 1000000000,
  "update_interval": 1,

  "1": {
    "candidate_steps": [3],
    "up_hysteresis": 0.0,
    "down_hysteresis": 0.0,
    "ceiling_coeff": 0
  },

  "32": {
    "candidate_steps": [0],
    "up_hysteresis": 0.0,
    "down_hysteresis": 0.0,
    "ceiling_coeff": 0
  }
}
JSON

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
echo "  Adaptive candidates:    [5]"
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

# export SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND=1
# export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

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
    --skip-server-warmup \
    --enable-metrics \
    --max-running-requests 256 \
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
