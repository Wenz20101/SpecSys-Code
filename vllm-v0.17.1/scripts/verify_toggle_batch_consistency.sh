#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://172.16.33.142:8001}"
MODEL="${MODEL:-Qwen3-14B}"
BATCH_SIZE="${BATCH_SIZE:-1}"
START_INDEX="${START_INDEX:-0}"
SEED="${SEED:-66666}"
TOGGLE_DELAY="${TOGGLE_DELAY:-0.5}"
OUT_DIR="${OUT_DIR:-/home/user/vllm/logs/toggle-consistency-batch-$(date +%Y%m%d-%H%M%S)}"
USE_CACHE_SALT="${USE_CACHE_SALT:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DATASET_PATH="${DATASET_PATH:-/home/user/dataset/mbpp/sanitized/test-00000-of-00001.parquet}"

mkdir -p "$OUT_DIR"

payload() {
    local label="$1"
    local index="$2"
    local sample_index=$((START_INDEX + index))
    local cache_salt="${OUT_DIR##*/}:${label}:${sample_index}:${SEED}"

    python3 - "$MODEL" "$DATASET_PATH" "$sample_index" \
        "$SEED" "$USE_CACHE_SALT" "$cache_salt" "$MAX_TOKENS" <<'PY'
import json
import sys
import pandas as pd

(
    model,
    dataset_path,
    sample_index,
    seed,
    use_cache_salt,
    cache_salt,
    max_tokens,
) = sys.argv[1:]

df = pd.read_parquet(dataset_path)
sample_index = int(sample_index)
if sample_index < 0 or sample_index >= len(df):
    raise SystemExit(
        f"sample_index {sample_index} is out of range for dataset size {len(df)}"
    )

sample = df.iloc[sample_index].to_dict()
prompt_text = sample.get("prompt") or sample.get("text") or next(iter(sample.values()))
tests = sample.get("test_list")

content = (
    "Write a python function to solve this MBPP task.\n\n"
    f"Task:\n{prompt_text}"
)
if tests is not None:
    content += f"\n\nTests:\n{tests}"

payload = {
    "model": model,
    "messages": [{"role": "user", "content": content}],
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "max_tokens": int(max_tokens),
    "seed": int(seed),
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}

if use_cache_salt == "1":
    payload["cache_salt"] = cache_salt

print(json.dumps(payload, ensure_ascii=False))
PY
}

write_prompt() {
    local index="$1"
    local prompt_path="$OUT_DIR/prompt_${index}.txt"
    local sample_index=$((START_INDEX + index))

    python3 - "$DATASET_PATH" "$sample_index" "$prompt_path" <<'PY'
import sys
import pandas as pd

dataset_path, sample_index, prompt_path = sys.argv[1:]

df = pd.read_parquet(dataset_path)
sample_index = int(sample_index)
sample = df.iloc[sample_index].to_dict()

prompt_text = sample.get("prompt") or sample.get("text") or next(iter(sample.values()))
tests = sample.get("test_list")

with open(prompt_path, "w", encoding="utf-8") as f:
    f.write(f"MBPP sample index: {sample_index}\n")
    if "task_id" in sample:
        f.write(f"task_id: {sample['task_id']}\n")
    f.write("\nTask:\n")
    f.write(str(prompt_text))
    if tests is not None:
        f.write("\n\nTests:\n")
        f.write(str(tests))
PY
}

toggle_sd() {
    local enabled="$1"

    curl -fsS -X POST "$BASE_URL/v1/engine/speculative_decoding" \
        -H "Content-Type: application/json" \
        -d "{\"enabled\": ${enabled}}" \
        > "$OUT_DIR/toggle-${enabled}-$(date +%s%N).json"
}

extract_text() {
    local json_path="$1"
    local text_path="$2"

    python3 - "$json_path" "$text_path" <<'PY'
import json
import sys

json_path, text_path = sys.argv[1:]

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

if "error" in data:
    raise SystemExit(f"API error in {json_path}: {data['error']}")

choice = data["choices"][0]
if "message" in choice:
    text = choice["message"].get("content", "")
else:
    text = choice.get("text", "")

with open(text_path, "w", encoding="utf-8") as f:
    f.write(text)
PY
}

request_one_async() {
    local label="$1"
    local index="$2"
    local json_path="$OUT_DIR/${label}_${index}.json"

    curl -fsS "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        --data-binary "$(payload "$label" "$index")" \
        > "$json_path" &
}

finish_batch() {
    local label="$1"
    shift
    local pids=("$@")
    local failures=0

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
    done

    if [ "$failures" -ne 0 ]; then
        echo "FAIL: ${failures} request(s) failed in ${label}"
        return 1
    fi

    for ((i = 0; i < BATCH_SIZE; i++)); do
        extract_text "$OUT_DIR/${label}_${i}.json" "$OUT_DIR/${label}_${i}.txt"
    done
}

run_batch() {
    local label="$1"
    local pids=()

    echo "launch batch: ${label} (size=${BATCH_SIZE})"

    for ((i = 0; i < BATCH_SIZE; i++)); do
        request_one_async "$label" "$i"
        pids+=("$!")
    done

    finish_batch "$label" "${pids[@]}"
}

run_batch_with_toggle() {
    local label="$1"
    local from_enabled="$2"
    local to_enabled="$3"
    local pids=()

    toggle_sd "$from_enabled"
    echo "launch batch: ${label} (size=${BATCH_SIZE}), then toggle ${from_enabled}->${to_enabled} after ${TOGGLE_DELAY}s"

    for ((i = 0; i < BATCH_SIZE; i++)); do
        request_one_async "$label" "$i"
        pids+=("$!")
    done

    sleep "$TOGGLE_DELAY"

    local running=0
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            running=$((running + 1))
        fi
    done

    echo "${running}/${BATCH_SIZE} request(s) still running; toggling SD to ${to_enabled}"
    toggle_sd "$to_enabled"

    finish_batch "$label" "${pids[@]}"
}

compare_one() {
    local expected_label="$1"
    local actual_label="$2"
    local index="$3"

    local expected="$OUT_DIR/${expected_label}_${index}.txt"
    local actual="$OUT_DIR/${actual_label}_${index}.txt"
    local diff_path="$OUT_DIR/diff-${expected_label}-vs-${actual_label}-${index}.txt"

    if cmp -s "$expected" "$actual"; then
        printf "PASS pair=%s_vs_%s request=%s sample_index=%s bytes=%s\n" \
            "$expected_label" \
            "$actual_label" \
            "$index" \
            "$((START_INDEX + index))" \
            "$(wc -c < "$expected")"
        return 0
    fi

    diff -u "$expected" "$actual" > "$diff_path" || true

    printf "FAIL pair=%s_vs_%s request=%s sample_index=%s expected_sha=%s actual_sha=%s diff=%s\n" \
        "$expected_label" \
        "$actual_label" \
        "$index" \
        "$((START_INDEX + index))" \
        "$(sha256sum "$expected" | awk '{print $1}')" \
        "$(sha256sum "$actual" | awk '{print $1}')" \
        "$diff_path"

    return 1
}

compare_batch() {
    local expected_label="$1"
    local actual_label="$2"
    local failures=0

    echo
    echo "compare: ${expected_label} vs ${actual_label}"

    for ((i = 0; i < BATCH_SIZE; i++)); do
        if ! compare_one "$expected_label" "$actual_label" "$i"; then
            failures=$((failures + 1))
        fi
    done

    if [ "$failures" -ne 0 ]; then
        echo "FAIL: ${failures}/${BATCH_SIZE} output(s) differed for ${expected_label} vs ${actual_label}"
        return 1
    fi

    echo "PASS: all ${BATCH_SIZE} outputs are identical for ${expected_label} vs ${actual_label}"
    return 0
}

compare_all() {
    local total_failures=0

    # 静态 Non-SD 与静态 SD 对比
    if ! compare_batch "baseline_nonsd" "baseline_sd"; then
        total_failures=$((total_failures + 1))
    fi

    # SD -> Non-SD：请求发起时是 SD，因此优先和 baseline_sd 对比
    if ! compare_batch "baseline_sd" "sd_to_nonsd"; then
        total_failures=$((total_failures + 1))
    fi

    # Non-SD -> SD：请求发起时是 Non-SD，因此优先和 baseline_nonsd 对比
    if ! compare_batch "baseline_nonsd" "nonsd_to_sd"; then
        total_failures=$((total_failures + 1))
    fi

    # 额外交叉对比，方便判断 toggle 是否改变了 in-flight request 的行为
    if ! compare_batch "baseline_nonsd" "sd_to_nonsd"; then
        total_failures=$((total_failures + 1))
    fi

    if ! compare_batch "baseline_sd" "nonsd_to_sd"; then
        total_failures=$((total_failures + 1))
    fi

    echo
    if [ "$total_failures" -ne 0 ]; then
        echo "FINAL: FAIL, ${total_failures} comparison group(s) have differences"
        return 1
    fi

    echo "FINAL: PASS, all comparison groups are identical"
    return 0
}

for ((i = 0; i < BATCH_SIZE; i++)); do
    write_prompt "$i"
done

echo "BASE_URL=$BASE_URL"
echo "MODEL=$MODEL"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "START_INDEX=$START_INDEX"
echo "SEED=$SEED"
echo "TOGGLE_DELAY=$TOGGLE_DELAY"
echo "MAX_TOKENS=$MAX_TOKENS"
echo "OUT_DIR=$OUT_DIR"
echo "DATASET_PATH=$DATASET_PATH"
echo "USE_CACHE_SALT=$USE_CACHE_SALT"

echo "1/4 force Non-SD baseline batch"
toggle_sd false
run_batch "baseline_nonsd"

echo "2/4 force SD baseline batch"
toggle_sd true
run_batch "baseline_sd"

echo "3/4 run batch, toggle SD -> Non-SD"
run_batch_with_toggle "sd_to_nonsd" true false

echo "4/4 run batch, toggle Non-SD -> SD"
run_batch_with_toggle "nonsd_to_sd" false true

compare_all

echo "All raw responses, prompts, extracted outputs, and diffs are saved in: $OUT_DIR"