# Tiny Model Pipeline

End-to-end: fine-tune → convert to GGUF → serve via llama.cpp → evaluate.

## Prerequisites

- GPU: NVIDIA with CUDA (RTX 3060 or better)
- Python venv: `.venv/` with torch, transformers, trl, peft, gguf, sentencepiece
- llama.cpp: `llama-server` binary at `~/.local/bin/llama-server`
- llama.cpp source: needed for `convert_hf_to_gguf.py` (clone to project dir)

### One-time setup

```bash
# Clone llama.cpp into project (not /tmp)
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git tools/llama.cpp

# Build with CUDA
cd tools/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Install binaries
cp build/bin/llama-server build/bin/llama-cli build/bin/llama-quantize ~/.local/bin/

# Install gguf python package
pip install tools/llama.cpp/gguf-py/
```

## Pipeline Steps

### Step 1: Train (14 min on RTX 3060)

Requires `mamba-ssm` and `causal-conv1d` packages (efficient Mamba2 CUDA kernels).
Without them, training OOMs on the naive torch fallback.

```bash
TMPDIR=/home/ubuntu/tmp pip install mamba-ssm causal-conv1d  # one-time, needs disk TMPDIR
.venv/bin/python train.py \
  --model granite-h-350m \
  --format b \
  --epochs 3 \
  --dataset-dir ./dataset/format_b_decisions
```

Output: `checkpoints/granite-h-350m_fmtb_full_ep3/final/`

The training script automatically copies the original model's config.json and
tokenizer files into the checkpoint directory. This is required because
transformers 5.x saves config.json in a format that llama.cpp's converter
doesn't handle.

### Step 2: Convert to GGUF (seconds)

```bash
.venv/bin/python tools/llama.cpp/convert_hf_to_gguf.py \
  checkpoints/granite-350m_fmtb_full_ep3/final/ \
  --outfile models/granite-350m-witness.gguf \
  --outtype f16
```

Output: `models/granite-350m-witness.gguf` (~705MB)

### Step 3: Serve

```bash
llama-server \
  --model models/granite-350m-witness.gguf \
  --host 127.0.0.1 --port 8081 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --jinja \
  --webui-mcp-proxy
```

WebUI: http://127.0.0.1:8081

### Step 4: Evaluate

```bash
# Via PyTorch (uses checkpoint directly)
.venv/bin/python evaluate.py --checkpoint checkpoints/granite-350m_fmtb_full_ep3/final

# Quick test via llama-server API
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a Witness agent. Respond ONLY with JSON: {\"tool\": \"<name>\", \"args\": {}}"},
      {"role": "user", "content": "Polecats\n\n  ● gastown/furiosa  working\n  ○ gastown/nux  done"}
    ],
    "max_tokens": 100, "temperature": 0
  }'
```

### Step 5: Push to HuggingFace

Model files are NOT stored in git. Push to HuggingFace:

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/granite-h-350m_fmtb_full_ep3/final',
    repo_id='dunks/granite-350m-witness',
    commit_message='Update model checkpoint',
)
"
```

## Model Registry

| Key | HuggingFace ID | Notes |
|-----|---------------|-------|
| `smollm2-135m` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 8K context, too small for general use |
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M-Instruct` | |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 32K context, standard transformer |
| `granite-350m` | `ibm-granite/granite-4.0-350m` | 32K context, hybrid Mamba2 |
| `granite-h-350m` | `ibm-granite/granite-4.0-h-350m` | 128K context, hybrid Mamba2, **current winner** |

## Known Issues

- **GGUF conversion requires original config.json**: transformers 5.x saves
  config.json with different key names (`dtype` vs `torch_dtype`, nested
  `rope_parameters` vs flat `rope_scaling`). The training script handles this
  automatically by copying original files after saving.

- **SmolLM2 GGUF**: Conversion produces garbage. Use the official GGUF from
  bartowski for base model, or serve via api_server.py (PyTorch).

## Files

| File | Purpose |
|------|---------|
| `train.py` | Fine-tuning (GPU required, bf16) |
| `evaluate.py` | 16-scenario eval harness |
| `api_server.py` | PyTorch OpenAI-compatible API server (fallback) |
| `serve.py` | Witness patrol shim (production) |
| `convert_gguf.py` | Wrapper around llama.cpp converter |
| `PIPELINE.md` | This file |
| `plans/model-candidates.md` | Model comparison and selection |
