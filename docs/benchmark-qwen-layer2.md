# Benchmarking Isartor Layer 2 with Qwen 2.5 Coder 7B

This runbook explains how to stand up a **real** Qwen 2.5 Coder 7B Instruct
sidecar via llama.cpp, wire it to Isartor's Layer 2 router, and validate
the setup end-to-end using the `claude_code_tasks` benchmark fixture.

---

## Model Details

| Field            | Value                                                         |
|------------------|---------------------------------------------------------------|
| **Model family** | Qwen 2.5 Coder (code-specialised)                            |
| **Size**         | 7B parameters                                                 |
| **Variant**      | Instruct (instruction-tuned for chat / completions)          |
| **Quantisation** | Q4_K_M — 4-bit, K-quant mixed, good quality/speed trade-off |
| **HF repo**      | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`                       |
| **HF filename**  | `qwen2.5-coder-7b-instruct-q4_k_m.gguf`                     |
| **Disk size**    | ≈ 4.7 GB                                                      |
| **Peak RAM**     | ≈ 5.5–6.5 GB (CPU inference)                                 |
| **Peak VRAM**    | ≈ 5.5 GB (full GPU offload on NVIDIA)                        |
| **Context**      | 8192 tokens (default in this setup; model supports 128 k)   |
| **Inference**    | llama.cpp server — OpenAI-compatible `/v1/chat/completions`  |

---

## Hardware Requirements

| Mode           | RAM     | VRAM  | CPU     | Notes                                          |
|----------------|---------|-------|---------|------------------------------------------------|
| CPU only       | 16 GB   | —     | 4 cores | ~3–8 tok/s on modern x86 — usable for validation |
| GPU (NVIDIA)   | 8 GB    | 8 GB  | 4 cores | ~30–60 tok/s on RTX 3060/3080; requires NVIDIA Container Toolkit |
| GPU (Apple M-series) | 8 GB unified | — | — | Run llama.cpp natively (not Docker); see note below |

> **Cost note**: A cloud VM with a T4 GPU (e.g. `g4dn.xlarge` on AWS, ≈ $0.50/hr)
> is sufficient for a single benchmark run. Stop the instance after validation
> to avoid ongoing charges.

---

## Quick Start (Docker Compose, CPU)

```bash
# 1. Clone the repository
git clone https://github.com/isartor-ai/Isartor.git
cd Isartor/docker

# 2. Copy and edit the environment file
cp .env.qwen-benchmark.example .env.qwen-benchmark
# (optional) edit .env.qwen-benchmark — set ISARTOR__GATEWAY_API_KEY if needed

# 3. Start the stack
#    First-run downloads qwen2.5-coder-7b-instruct-q4_k_m.gguf (~4.7 GB).
#    Subsequent starts use the cached `isartor-qwen-model-cache` volume.
docker compose -f docker-compose.qwen-benchmark.yml up --build

# 4. Wait for both services to become healthy (allow up to 10 min for first-run download)
docker compose -f docker-compose.qwen-benchmark.yml ps
# Expected: qwen-sidecar = healthy, isartor-qwen-gateway = healthy
```

---

## GPU Passthrough (NVIDIA)

### 1. Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Set `LLAMA_GPU_LAYERS` in your env file

Edit `docker/.env.qwen-benchmark`:

```bash
LLAMA_GPU_LAYERS=33   # Qwen 2.5 7B has 33 transformer layers → full GPU offload
```

### 3. Add a GPU resource reservation override

Create `docker/docker-compose.gpu.override.yml` (if it does not already exist):

```yaml
services:
  qwen-sidecar:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 4. Start with the GPU override

```bash
cd docker
docker compose \
  -f docker-compose.qwen-benchmark.yml \
  -f docker-compose.gpu.override.yml \
  up --build
```

---

## Smoke Tests

### Health checks

```bash
# Isartor gateway
curl -sf http://localhost:8080/healthz && echo "gateway OK"

# llama.cpp sidecar
curl -sf http://localhost:8081/health && echo "sidecar OK"
```

### Direct sidecar inference (Layer 2 only)

Send a coding question directly to the sidecar to confirm the model is loaded
and responding correctly:

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function that checks if a number is prime."
      }
    ],
    "max_tokens": 256,
    "temperature": 0
  }' | jq -r '.choices[0].message.content'
```

Expected: a short, correct Python function. If the sidecar returns an error or
the output is garbled, the model may still be loading — wait a moment and retry.

### Isartor Layer 2 routing (end-to-end)

Send a request through Isartor and confirm `X-Isartor-Layer: l2`:

```bash
curl -s http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme" \
  -d '{"prompt": "Write a Python function that checks if a number is prime."}' \
  -i | grep -E "X-Isartor-Layer|X-Isartor-Deflected"
```

Expected output:

```
X-Isartor-Layer: l2
X-Isartor-Deflected: true
```

A `l1a` or `l1b` hit on a warm cache is also a deflection success. The first
request for a new prompt will reach `l2` because the cache is cold.

---

## Running the Claude Code Benchmark Fixture

The `claude_code_tasks` fixture contains 388 diverse coding prompts that
represent typical Claude Code / GitHub Copilot workloads. It is designed to
stress Layer 2 with questions the SLM can answer locally.

```bash
# From the repository root (server must be running on localhost:8080)
python3 benchmarks/run.py \
  --url http://localhost:8080 \
  --api-key changeme \
  --input benchmarks/fixtures/claude_code_tasks.jsonl \
  --timeout 180

# Makefile shortcut (uses ISARTOR_URL / ISARTOR_API_KEY env vars)
make benchmark-qwen
```

### Expected result shape

```
-- claude_code_tasks --
  Total requests :  388
  L1a (exact)    :   0  ( 0.0%)   ← cold cache, first run
  L1b (semantic) :   0  ( 0.0%)   ← cold cache, first run
  L2  (SLM)      : ~230 (~60.0%)  ← Qwen 2.5 Coder resolves most coding Qs
  L3  (cloud)    : ~158 (~40.0%)  ← complex or novel questions fall through
  Deflection rate: ~60%
```

> Results will vary depending on hardware, GPU layers, and Layer 3 configuration.
> On CPU-only the sidecar is slower but the deflection rate should be similar.
> Cache hits (L1a/L1b) accumulate on repeated runs.

---

## Environment Variables Reference

All variables are loaded from `docker/.env.qwen-benchmark` (or from the shell
environment). Double-underscore separators map to nested config fields.

| Variable                              | Default                          | Description                                          |
|---------------------------------------|----------------------------------|------------------------------------------------------|
| `ISARTOR__GATEWAY_API_KEY`            | `changeme`                       | Authentication key for Isartor API                  |
| `ISARTOR__ENABLE_SLM_ROUTER`          | `true` (hardcoded in compose)    | Enables Layer 2 SLM routing                         |
| `ISARTOR__LAYER2__SIDECAR_URL`        | `http://qwen-sidecar:8081`       | URL of the llama.cpp sidecar                        |
| `ISARTOR__LAYER2__MODEL_NAME`         | `qwen2.5-coder-7b`               | Model name sent in OpenAI API requests              |
| `ISARTOR__LAYER2__TIMEOUT_SECONDS`    | `120`                            | Per-request timeout for sidecar calls               |
| `LLAMA_GPU_LAYERS`                    | `0`                              | Number of transformer layers to offload to GPU      |
| `ISARTOR__LLM_PROVIDER`               | `openai`                         | Layer 3 provider (`openai`, `azure`, `copilot`, …)  |
| `ISARTOR__EXTERNAL_LLM_API_KEY`       | *(empty — disables Layer 3)*     | API key for Layer 3 cloud fallback                  |

---

## Tearing Down

```bash
cd docker
docker compose -f docker-compose.qwen-benchmark.yml down

# Remove the downloaded model weights (frees ~4.7 GB of disk)
docker volume rm isartor-qwen-model-cache
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `qwen-sidecar` stuck in `starting` for > 10 min | Model still downloading | Check logs: `docker compose logs qwen-sidecar` |
| Sidecar returns 503 | Model not yet loaded into memory | Wait for `llama server listening` in sidecar logs |
| All requests return `l3` instead of `l2` | `ISARTOR__ENABLE_SLM_ROUTER` is `false` | Verify env file is loaded; check gateway logs |
| `X-Isartor-Layer: l1a` on first request | Exact cache hit from a previous run | Normal behaviour on warm cache; clear cache by restarting gateway |
| Out of memory / OOM kill on qwen-sidecar | Insufficient RAM/VRAM | Reduce `--ctx-size` to `4096` or use a smaller quantisation (Q3_K_M) |
| Hugging Face download fails | Rate limit or network restriction | Set `HF_TOKEN` env var on `qwen-sidecar` for authenticated download |

---

## Apple Silicon (Native llama.cpp)

Docker does not expose Metal/MPS GPU on macOS.  For Apple M-series hardware,
run llama.cpp natively for GPU acceleration:

```bash
# Install via Homebrew (builds with Metal support)
brew install llama.cpp

# Download the GGUF model
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --local-dir ~/models/qwen

# Start the server
llama-server \
  --model ~/models/qwen/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8081 \
  --ctx-size 8192 \
  --n-gpu-layers 33 \
  --chat-template chatml
```

Then configure Isartor with:

```bash
ISARTOR__ENABLE_SLM_ROUTER=true
ISARTOR__LAYER2__SIDECAR_URL=http://localhost:8081
ISARTOR__LAYER2__MODEL_NAME=qwen2.5-coder-7b
ISARTOR__LAYER2__TIMEOUT_SECONDS=120
```

---

## See Also

- [`docker/docker-compose.qwen-benchmark.yml`](../../docker/docker-compose.qwen-benchmark.yml) — compose file for this setup
- [`docker/.env.qwen-benchmark.example`](../../docker/.env.qwen-benchmark.example) — environment template
- [`benchmarks/fixtures/claude_code_tasks.jsonl`](../../benchmarks/fixtures/claude_code_tasks.jsonl) — benchmark fixture
- [`docs/deploy-level2-sidecar.md`](deploy-level2-sidecar.md) — general Level 2 sidecar deployment guide
- [Qwen 2.5 Coder GGUF on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
