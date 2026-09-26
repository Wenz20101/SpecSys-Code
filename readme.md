# SpecSys-Code Quick Start

This repository contains three components:

- `scheduler/`: The ThunderAgent request proxy and scheduler, which can dynamically switch SD based on the current load.
- `sglang-v0.5.14/`: A modified SGLang version that supports runtime SD/Non-SD switching.
- `vllm-v0.17.1/`: A modified vLLM version that supports iteration-level SD/Non-SD switching.

SD means Speculative Decoding. The commands below assume a Linux system with NVIDIA GPUs and CUDA.

## 1. Environment Setup

SGLang and vLLM require different PyTorch versions. Always install them in separate environments.

### Scheduler

```bash
cd /home/user/SpecSys-Code
conda create -n specsys-scheduler python=3.12 -y
conda activate specsys-scheduler
python -m pip install -U pip
python -m pip install -e ./scheduler

# Verify the installation.
thunderagent --help
```

### SGLang v0.5.14

`sglang-v0.5.14/python/pyproject.toml` requires Python 3.10 or newer and pins `torch==2.11.0` and CUDA 13 dependencies.

```bash
cd /home/user/SpecSys-Code
conda create -n sglang-0514 python=3.12 -y
conda activate sglang-0514
python -m pip install -U pip uv
cd sglang-v0.5.14
uv pip install -e ./python

# Verify the installation.
python -c "import sglang; print(sglang.__file__)"
```

### vLLM v0.17.1

vLLM requires Python 3.10 or newer and earlier than 3.14. Its build dependencies use `torch==2.10.0`. The editable source installation below ensures that the SD-switching changes in this repository are used.

```bash
cd /home/user/SpecSys-Code
conda create -n vllm-0171 python=3.12 -y
conda activate vllm-0171
python -m pip install -U pip uv
cd vllm-v0.17.1
uv pip install -e .

# Verify the installation.
python -c "import vllm; print(vllm.__file__)"
```

If you only modify Python code and have a precompiled vLLM wheel compatible with the current source, use the following faster installation method:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

## 2. Start an SD-Capable Backend

### SGLang

First, edit `MODEL`, `DRAFT_MODEL`, `GPU`, `TP`, and `PORT` in `sglang-v0.5.14/scripts/launch_sglang_sd.sh`. The `LOG_FILE` and `PID_FILE` variables still use old absolute paths; update them to paths under `/home/user/SpecSys-Code/sglang-v0.5.14/` as well.

```bash
cd /home/user/SpecSys-Code/sglang-v0.5.14
bash scripts/launch_sglang_sd.sh
```

This script starts the server in EAGLE3 SD mode. To run a pure Non-SD baseline without runtime switching support, update the corresponding settings and run:

```bash
bash scripts/launch_sglang_nonsd.sh
```

### vLLM

Runtime switching is available only when vLLM is started with `--speculative-config`. The following example uses EAGLE3. Replace the model paths and GPU settings as needed:

```bash
conda activate vllm-0171
cd /home/user/SpecSys-Code/vllm-v0.17.1

CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/target-model \
  --served-model-name Qwen3-14B \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 2 \
  --speculative-config '{"model":"/path/to/eagle3-draft-model","method":"eagle3","num_speculative_tokens":3,"draft_tensor_parallel_size":2}'
```

## 3. Manually Switch SD / Non-SD

SGLang and vLLM expose the same HTTP endpoint. Assuming that the backend is running at `http://127.0.0.1:8001`:

```bash
# Enable SD.
curl -sS -X POST http://127.0.0.1:8001/v1/engine/speculative_decoding \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true}'

# Disable SD and switch to Non-SD.
curl -sS -X POST http://127.0.0.1:8001/v1/engine/speculative_decoding \
  -H 'Content-Type: application/json' \
  -d '{"enabled":false}'
```

The new state takes effect at the next scheduler iteration and does not interrupt an iteration that is already running. A successful response contains `"status": "success"`. If the engine was started without an SD configuration, the response status is `ignored`.

SGLang runtime switching currently supports only EAGLE/EAGLE3 with `speculative_eagle_topk=1`.

## 4. Automatically Switch SD with the Scheduler

Start an SD-capable SGLang or vLLM backend first, then start ThunderAgent. The following example uses a vLLM backend:

```bash
conda activate specsys-scheduler
cd /home/user/SpecSys-Code/scheduler

thunderagent \
  --host 0.0.0.0 \
  --port 9001 \
  --backends http://127.0.0.1:8001 \
  --backend-type vllm \
  --metrics \
  --dynamic-sd \
  --sd-switch-threshold 64
```

For SGLang, change `--backend-type vllm` to `--backend-type sglang`. Multiple backend URLs can be supplied as a comma-separated list to `--backends`.

The scheduler disables SD when the projected next running batch size reaches `--sd-switch-threshold`. It enables SD again when the load falls to 75% of that threshold or lower. For example, with a threshold of 64, SD is disabled at 64 and re-enabled at 48.

After ThunderAgent starts, send OpenAI-compatible requests to the proxy port. The following commands can be used to check the service:

```bash
curl http://127.0.0.1:9001/health
curl http://127.0.0.1:9001/v1/models
```

To run without automatic SD switching, omit `--dynamic-sd` and `--sd-switch-threshold`.

The three command blocks in `scheduler/launch.sh` are separate scheduling examples. Select only one of them; do not start all three at the same time.
