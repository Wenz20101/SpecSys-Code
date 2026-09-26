#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://172.16.33.142:8001}"
MODEL="${MODEL:-Qwen3-14B}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_RUNS="${NUM_RUNS:-5}"
START_INDEX="${START_INDEX:-0}"
SEED="${SEED:-66666}"
RUN_GAP="${RUN_GAP:-1}"
USE_CACHE_SALT="${USE_CACHE_SALT:-0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DATASET_PATH="${DATASET_PATH:-/home/user/dataset/mbpp/sanitized/test-00000-of-00001.parquet}"
LABEL="${LABEL:-specsys-sd}"
OUT_DIR="${OUT_DIR:-/home/user/vllm/logs/${LABEL}_b${BATCH_SIZE}_speed_tp2}"

mkdir -p "$OUT_DIR"

METRICS_TSV="$OUT_DIR/metrics.tsv"
SUMMARY_TSV="$OUT_DIR/summary.tsv"
AGGREGATE_TSV="$OUT_DIR/aggregate_summary.tsv"

printf "run_id\tlabel\trequest_index\tsample_index\tstart_ns\tend_ns\tjct_s\tstatus\thttp_code\tjson_path\n" > "$METRICS_TSV"
printf "run_id\tlabel\tbatch_size\tok_count\tfailed_count\tbatch_wall_s\tavg_jct_s\tmin_jct_s\tp50_jct_s\tp90_jct_s\tp95_jct_s\tmax_jct_s\tthroughput_jobs_per_s\n" > "$SUMMARY_TSV"
printf "label\tnum_runs\tbatch_size\ttotal_ok\ttotal_failed\tmean_batch_wall_s\tmean_avg_jct_s\tmean_p50_jct_s\tmean_p90_jct_s\tmean_p95_jct_s\tmean_throughput_jobs_per_s\tpooled_avg_jct_s\toverall_throughput_jobs_per_s\n" > "$AGGREGATE_TSV"

write_payload() {
    local run_id="$1"
    local label="$2"
    local index="$3"

    local sample_index=$((START_INDEX + index))
    local cache_salt="${OUT_DIR##*/}:run${run_id}:${label}:${sample_index}:${SEED}"
    local payload_path="$OUT_DIR/payload_run${run_id}_${label}_${index}.json"

    python3 - "$MODEL" "$DATASET_PATH" "$sample_index" \
        "$SEED" "$USE_CACHE_SALT" "$cache_salt" "$MAX_TOKENS" "$payload_path" <<'PY'
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
    payload_path,
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

with open(payload_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)
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

prepare_payloads() {
    local run_id="$1"
    local label="$2"

    echo "prepare payloads: run=${run_id}, label=${label}"

    for ((i = 0; i < BATCH_SIZE; i++)); do
        write_payload "$run_id" "$label" "$i"
    done
}

request_one_async() {
    local run_id="$1"
    local label="$2"
    local index="$3"

    local sample_index=$((START_INDEX + index))
    local payload_path="$OUT_DIR/payload_run${run_id}_${label}_${index}.json"
    local json_path="$OUT_DIR/run${run_id}_${label}_${index}.json"
    local metric_path="$OUT_DIR/metric_run${run_id}_${label}_${index}.tsv"

    (
        set +e

        start_ns="$(date +%s%N)"

        http_code="$(
            curl -sS \
                -o "$json_path" \
                -w "%{http_code}" \
                "$BASE_URL/v1/chat/completions" \
                -H "Content-Type: application/json" \
                --data-binary @"$payload_path"
        )"

        curl_status="$?"
        end_ns="$(date +%s%N)"

        jct_s="$(
            python3 - "$start_ns" "$end_ns" <<'PY'
import sys

start_ns, end_ns = map(int, sys.argv[1:])
print(f"{(end_ns - start_ns) / 1e9:.6f}")
PY
        )"

        status=0

        if [ "$curl_status" -ne 0 ]; then
            status=1
        elif ! [[ "$http_code" =~ ^[0-9]{3}$ ]]; then
            status=1
        elif (( http_code < 200 || http_code >= 300 )); then
            status=1
        else
            if ! python3 - "$json_path" <<'PY'
import json
import sys

json_path = sys.argv[1]

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

if "error" in data:
    raise SystemExit(f"API error: {data['error']}")

if "choices" not in data or not data["choices"]:
    raise SystemExit("missing choices in response")
PY
            then
                status=2
            fi
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$run_id" \
            "$label" \
            "$index" \
            "$sample_index" \
            "$start_ns" \
            "$end_ns" \
            "$jct_s" \
            "$status" \
            "$http_code" \
            "$json_path" \
            > "$metric_path"

        exit "$status"
    ) &
}

summarize_batch() {
    local run_id="$1"
    local label="$2"

    python3 - "$OUT_DIR" "$run_id" "$label" "$BATCH_SIZE" "$METRICS_TSV" "$SUMMARY_TSV" <<'PY'
import csv
import math
import os
import sys

out_dir, run_id, label, batch_size, metrics_tsv, summary_tsv = sys.argv[1:]
batch_size = int(batch_size)

rows = []

for i in range(batch_size):
    path = os.path.join(out_dir, f"metric_run{run_id}_{label}_{i}.tsv")

    if not os.path.exists(path):
        rows.append({
            "run_id": run_id,
            "label": label,
            "request_index": str(i),
            "sample_index": "",
            "start_ns": "0",
            "end_ns": "0",
            "jct_s": "nan",
            "status": "99",
            "http_code": "000",
            "json_path": "",
        })
        continue

    with open(path, encoding="utf-8") as f:
        line = f.readline().rstrip("\n")
        parts = line.split("\t")

    rows.append({
        "run_id": parts[0],
        "label": parts[1],
        "request_index": parts[2],
        "sample_index": parts[3],
        "start_ns": parts[4],
        "end_ns": parts[5],
        "jct_s": parts[6],
        "status": parts[7],
        "http_code": parts[8],
        "json_path": parts[9],
    })

rows.sort(key=lambda r: int(r["request_index"]))

with open(metrics_tsv, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    for r in rows:
        writer.writerow([
            r["run_id"],
            r["label"],
            r["request_index"],
            r["sample_index"],
            r["start_ns"],
            r["end_ns"],
            r["jct_s"],
            r["status"],
            r["http_code"],
            r["json_path"],
        ])

ok_rows = [r for r in rows if r["status"] == "0"]
failed_count = batch_size - len(ok_rows)

timed_rows = [
    r for r in rows
    if r["start_ns"].isdigit()
    and r["end_ns"].isdigit()
    and int(r["start_ns"]) > 0
    and int(r["end_ns"]) > 0
]

if timed_rows:
    batch_start_ns = min(int(r["start_ns"]) for r in timed_rows)
    batch_end_ns = max(int(r["end_ns"]) for r in timed_rows)
    batch_wall_s = (batch_end_ns - batch_start_ns) / 1e9
else:
    batch_wall_s = float("nan")

jcts = [float(r["jct_s"]) for r in ok_rows]

def fmt(x):
    if x is None or math.isnan(x):
        return "nan"
    return f"{x:.6f}"

def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = math.ceil(len(values) * p / 100.0) - 1
    k = max(0, min(k, len(values) - 1))
    return values[k]

if jcts:
    avg_jct_s = sum(jcts) / len(jcts)
    min_jct_s = min(jcts)
    p50_jct_s = percentile(jcts, 50)
    p90_jct_s = percentile(jcts, 90)
    p95_jct_s = percentile(jcts, 95)
    max_jct_s = max(jcts)
else:
    avg_jct_s = min_jct_s = p50_jct_s = p90_jct_s = p95_jct_s = max_jct_s = float("nan")

if batch_wall_s and not math.isnan(batch_wall_s) and batch_wall_s > 0:
    throughput_jobs_per_s = len(ok_rows) / batch_wall_s
else:
    throughput_jobs_per_s = float("nan")

with open(summary_tsv, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        run_id,
        label,
        batch_size,
        len(ok_rows),
        failed_count,
        fmt(batch_wall_s),
        fmt(avg_jct_s),
        fmt(min_jct_s),
        fmt(p50_jct_s),
        fmt(p90_jct_s),
        fmt(p95_jct_s),
        fmt(max_jct_s),
        fmt(throughput_jobs_per_s),
    ])

print(
    f"SUMMARY run={run_id} "
    f"label={label} "
    f"ok={len(ok_rows)}/{batch_size} "
    f"failed={failed_count} "
    f"batch_wall_s={fmt(batch_wall_s)} "
    f"avg_jct_s={fmt(avg_jct_s)} "
    f"throughput_jobs_per_s={fmt(throughput_jobs_per_s)}"
)
PY
}

finish_batch() {
    local run_id="$1"
    local label="$2"
    shift 2

    local pids=("$@")
    local failures=0

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
    done

    summarize_batch "$run_id" "$label"

    if [ "$failures" -ne 0 ]; then
        echo "FAIL: ${failures} request(s) failed in run=${run_id}, label=${label}"
        return 1
    fi

    return 0
}

run_batch() {
    local run_id="$1"
    local label="$2"
    local pids=()

    prepare_payloads "$run_id" "$label"

    echo
    echo "launch pure Non-SD batch: run=${run_id}/${NUM_RUNS}, label=${label}, size=${BATCH_SIZE}"

    for ((i = 0; i < BATCH_SIZE; i++)); do
        request_one_async "$run_id" "$label" "$i"
        pids+=("$!")
    done

    finish_batch "$run_id" "$label" "${pids[@]}"
}

aggregate_summary() {
    python3 - "$SUMMARY_TSV" "$METRICS_TSV" "$AGGREGATE_TSV" "$LABEL" "$BATCH_SIZE" <<'PY'
import csv
import math
import sys

summary_tsv, metrics_tsv, aggregate_tsv, label, batch_size = sys.argv[1:]
batch_size = int(batch_size)

summary_rows = []

with open(summary_tsv, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["label"] == label:
            summary_rows.append(row)

def to_float(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan")
    return sum(values) / len(values)

def fmt(x):
    if x is None or math.isnan(x):
        return "nan"
    return f"{x:.6f}"

num_runs = len(summary_rows)
total_ok = sum(int(r["ok_count"]) for r in summary_rows)
total_failed = sum(int(r["failed_count"]) for r in summary_rows)

mean_batch_wall_s = mean([to_float(r["batch_wall_s"]) for r in summary_rows])
mean_avg_jct_s = mean([to_float(r["avg_jct_s"]) for r in summary_rows])
mean_p50_jct_s = mean([to_float(r["p50_jct_s"]) for r in summary_rows])
mean_p90_jct_s = mean([to_float(r["p90_jct_s"]) for r in summary_rows])
mean_p95_jct_s = mean([to_float(r["p95_jct_s"]) for r in summary_rows])
mean_throughput_jobs_per_s = mean([to_float(r["throughput_jobs_per_s"]) for r in summary_rows])

all_jcts = []
with open(metrics_tsv, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["label"] == label and row["status"] == "0":
            v = to_float(row["jct_s"])
            if v is not None:
                all_jcts.append(v)

if all_jcts:
    pooled_avg_jct_s = sum(all_jcts) / len(all_jcts)
else:
    pooled_avg_jct_s = float("nan")

sum_batch_wall_s = sum(
    v for v in [to_float(r["batch_wall_s"]) for r in summary_rows]
    if v is not None
)

if sum_batch_wall_s > 0:
    overall_throughput_jobs_per_s = total_ok / sum_batch_wall_s
else:
    overall_throughput_jobs_per_s = float("nan")

with open(aggregate_tsv, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        label,
        num_runs,
        batch_size,
        total_ok,
        total_failed,
        fmt(mean_batch_wall_s),
        fmt(mean_avg_jct_s),
        fmt(mean_p50_jct_s),
        fmt(mean_p90_jct_s),
        fmt(mean_p95_jct_s),
        fmt(mean_throughput_jobs_per_s),
        fmt(pooled_avg_jct_s),
        fmt(overall_throughput_jobs_per_s),
    ])

print()
print("AGGREGATE SUMMARY")
print(
    f"label={label} "
    f"num_runs={num_runs} "
    f"batch_size={batch_size} "
    f"total_ok={total_ok} "
    f"total_failed={total_failed} "
    f"mean_avg_jct_s={fmt(mean_avg_jct_s)} "
    f"mean_throughput_jobs_per_s={fmt(mean_throughput_jobs_per_s)} "
    f"pooled_avg_jct_s={fmt(pooled_avg_jct_s)} "
    f"overall_throughput_jobs_per_s={fmt(overall_throughput_jobs_per_s)}"
)
PY
}

for ((i = 0; i < BATCH_SIZE; i++)); do
    write_prompt "$i"
done

echo "BASE_URL=$BASE_URL"
echo "MODEL=$MODEL"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_RUNS=$NUM_RUNS"
echo "START_INDEX=$START_INDEX"
echo "SEED=$SEED"
echo "MAX_TOKENS=$MAX_TOKENS"
echo "RUN_GAP=$RUN_GAP"
echo "OUT_DIR=$OUT_DIR"
echo "DATASET_PATH=$DATASET_PATH"
echo "USE_CACHE_SALT=$USE_CACHE_SALT"
echo "LABEL=$LABEL"

for ((run_id = 1; run_id <= NUM_RUNS; run_id++)); do
    run_batch "$run_id" "$LABEL"

    if [ "$run_id" -lt "$NUM_RUNS" ]; then
        echo "sleep ${RUN_GAP}s before next run"
        sleep "$RUN_GAP"
    fi
done

aggregate_summary

echo
echo "PER-RUN SUMMARY:"
column -t -s $'\t' "$SUMMARY_TSV" || cat "$SUMMARY_TSV"

echo
echo "AGGREGATE SUMMARY:"
column -t -s $'\t' "$AGGREGATE_TSV" || cat "$AGGREGATE_TSV"

echo
echo "Per-request metrics saved in: $METRICS_TSV"
echo "Per-run summary saved in: $SUMMARY_TSV"
echo "Aggregate summary saved in: $AGGREGATE_TSV"
echo "All raw responses, prompts, payloads, and metrics are saved in: $OUT_DIR"