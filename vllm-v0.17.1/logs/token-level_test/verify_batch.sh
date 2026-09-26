#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-172.16.33.142}"
NON_SD_BASE_URL="${NON_SD_BASE_URL:-http://${HOST}:8001}"
SD_BASE_URL="${SD_BASE_URL:-http://${HOST}:8002}"
MODEL="${MODEL:-Qwen3-14B}"
BATCH_SIZE="${BATCH_SIZE:-4}"
START_INDEX="${START_INDEX:-0}"
SEED="${SEED:-66666}"
OUT_DIR="${OUT_DIR:-/home/user/vllm/logs/vllm-dsd-vs-sd-b${BATCH_SIZE}}"
USE_CACHE_SALT="${USE_CACHE_SALT:-1}"
TOGGLE_SD_TO_NONSD="${TOGGLE_SD_TO_NONSD:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DATASET_PATH="${DATASET_PATH:-/home/user/dataset/mbpp/sanitized/test-00000-of-00001.parquet}"

mkdir -p "$OUT_DIR"

payload() {
    local label="$1"
    local index="$2"
    local sample_index=$((START_INDEX + index))
    local cache_salt="${OUT_DIR##*/}:${label}:${sample_index}:${SEED}"

    python3 - "$MODEL" "$SEED" "$DATASET_PATH" "$sample_index" \
        "$USE_CACHE_SALT" "$cache_salt" "$MAX_TOKENS" <<'PY'
import json
import sys
import pandas as pd

(
    model,
    seed,
    dataset_path,
    sample_index,
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
if "text" in sample:
    prompt_text = sample["text"]
elif "prompt" in sample:
    prompt_text = sample["prompt"]
else:
    prompt_text = next(iter(sample.values()))

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": f"Write a python function to solve this: {prompt_text}",
        }
    ],
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "max_tokens": int(max_tokens),
    "seed": int(seed),
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "chat_template_kwargs": {
        "enable_thinking": False,
    },
}

if use_cache_salt == "1":
    payload["cache_salt"] = cache_salt

print(json.dumps(payload, ensure_ascii=False))
PY
}

write_prompt() {
    local index="$1"
    local sample_index=$((START_INDEX + index))
    local prompt_path="$OUT_DIR/prompt_${index}.txt"

    python3 - "$DATASET_PATH" "$sample_index" "$prompt_path" <<'PY'
import sys
import pandas as pd

dataset_path, sample_index, prompt_path = sys.argv[1:]
df = pd.read_parquet(dataset_path)
sample_index = int(sample_index)
sample = df.iloc[sample_index].to_dict()
if "text" in sample:
    prompt_text = sample["text"]
elif "prompt" in sample:
    prompt_text = sample["prompt"]
else:
    prompt_text = next(iter(sample.values()))

with open(prompt_path, "w", encoding="utf-8") as f:
    f.write(f"sample_index: {sample_index}\n")
    if "task_id" in sample:
        f.write(f"task_id: {sample['task_id']}\n")
    f.write("\n")
    f.write(str(prompt_text))
PY
}

toggle_sd() {
    local enabled="$1"

    # curl -fsS -X POST "$SD_BASE_URL/v1/engine/speculative_decoding" \
    #     -H "Content-Type: application/json" \
    #     -d "{\"enabled\": ${enabled}}" \
    #     > "$OUT_DIR/toggle-${enabled}.json"
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
    local base_url="$2"
    local index="$3"
    local json_path="$OUT_DIR/${label}_${index}.json"

    curl -fsS "$base_url/v1/chat/completions" \
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
    local base_url="$2"
    local pids=()

    echo "launch batch: ${label} (size=${BATCH_SIZE}, url=${base_url})"
    for ((i = 0; i < BATCH_SIZE; i++)); do
        request_one_async "$label" "$base_url" "$i"
        pids+=("$!")
    done

    finish_batch "$label" "${pids[@]}"
}

compare_one() {
    local index="$1"
    local expected="$OUT_DIR/baseline_nonsd_${index}.txt"
    local actual="$OUT_DIR/sd_as_nonsd_${index}.txt"
    local diff_path="$OUT_DIR/diff-${index}.txt"

    if cmp -s "$expected" "$actual"; then
        printf "PASS request=%s sample_index=%s bytes=%s\n" \
            "$index" "$((START_INDEX + index))" "$(wc -c < "$expected")"
        return 0
    fi

    diff -u "$expected" "$actual" > "$diff_path" || true
    printf "FAIL request=%s sample_index=%s expected_sha=%s actual_sha=%s diff=%s\n" \
        "$index" \
        "$((START_INDEX + index))" \
        "$(sha256sum "$expected" | awk '{print $1}')" \
        "$(sha256sum "$actual" | awk '{print $1}')" \
        "$diff_path"
    return 1
}

compare_batch() {
    local failures=0

    for ((i = 0; i < BATCH_SIZE; i++)); do
        compare_one "$i" || failures=$((failures + 1))
    done

    if [ "$failures" -ne 0 ]; then
        echo "FAIL: ${failures}/${BATCH_SIZE} output(s) differed"
        return 1
    fi

    echo "PASS: all ${BATCH_SIZE} outputs are identical"
}

for ((i = 0; i < BATCH_SIZE; i++)); do
    write_prompt "$i"
done

echo "NON_SD_BASE_URL=$NON_SD_BASE_URL"
echo "SD_BASE_URL=$SD_BASE_URL"
echo "MODEL=$MODEL"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "START_INDEX=$START_INDEX"
echo "SEED=$SEED"
echo "MAX_TOKENS=$MAX_TOKENS"
echo "OUT_DIR=$OUT_DIR"
echo "DATASET_PATH=$DATASET_PATH"
echo "USE_CACHE_SALT=$USE_CACHE_SALT"
echo "TOGGLE_SD_TO_NONSD=$TOGGLE_SD_TO_NONSD"

if [ "$TOGGLE_SD_TO_NONSD" = "1" ]; then
    echo "1/3 force SD engine to Non-SD mode"
    toggle_sd true
else
    echo "1/3 skip SD toggle"
fi

echo "2/3 request pure Non-SD engine batch"
run_batch "baseline_nonsd" "$NON_SD_BASE_URL"

echo "3/3 request SD engine in Non-SD mode batch"
run_batch "sd_as_nonsd" "$SD_BASE_URL"

compare_batch

echo "All raw responses, prompts, extracted outputs, and diffs are saved in: $OUT_DIR"
