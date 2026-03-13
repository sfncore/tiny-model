# Tiny Model — Quick Start

Fine-tune a sub-500M LLM to replace the Gas Town witness agent.

## Setup

```bash
git clone git@github.com:sfncore/tiny-model.git
cd tiny-model
pip install torch transformers trl peft datasets accelerate
```

## Train

Dataset is included — no collection step needed.

```bash
# SmolLM2-135M (default, fastest)
python train.py --model smollm2-135m --format b --epochs 3 --dataset-dir ./dataset/format_b_decisions

# SmolLM2-360M (better quality, still fast on GPU)
python train.py --model smollm2-360m --format b --epochs 3 --dataset-dir ./dataset/format_b_decisions

# Qwen3-0.6B (largest supported)
python train.py --model qwen3-0.6b --format b --epochs 3 --dataset-dir ./dataset/format_b_decisions
```

### LoRA variant (uses less memory)

```bash
python train.py --model smollm2-135m --format b --epochs 3 --lora --dataset-dir ./dataset/format_b_decisions
```

### Training time estimates

| Model | CPU (8-core) | GPU (3060) |
|-------|-------------|------------|
| smollm2-135m | ~2.5h | ~5min |
| smollm2-360m | ~6h | ~15min |
| qwen3-0.6b | ~10h | ~25min |

## Evaluate

After training completes, checkpoints land in `./checkpoints/<run_name>/final/`.

```bash
python evaluate.py --model-path ./checkpoints/smollm2-135m_fmtb_full_ep3/final
```

Tests 8 patrol scenarios: idle, healthy polecat, stuck polecat, completed polecat, crash loop, infrastructure down, help request, mail processing.

## Serve

Run the inference shim for witness patrol integration:

```bash
python serve.py --model-path ./checkpoints/smollm2-135m_fmtb_full_ep3/final
```

## Dataset

- `dataset/format_b_decisions/format_b/train.jsonl` — 485 examples (185 real + 300 synthetic)
- `dataset/format_b_decisions/format_b/eval.jsonl` — 22 examples

### Regenerate from scratch

If you want to rebuild the dataset from raw session data:

```bash
python collect_training_data.py          # Step 1: export raw sessions
python curate.py --output-dir ./dataset/curated  # Step 2: filter quality
python extract_decisions.py              # Step 3: extract decision pairs
python synthetic_scenarios.py --n 300 --format both  # Step 4: synthetic data
# Then merge train.jsonl + synthetic_scenarios.jsonl manually
```

## Files

| File | Purpose |
|------|---------|
| `train.py` | SFT training (full fine-tune or LoRA) |
| `evaluate.py` | 8-scenario evaluation harness |
| `serve.py` | Inference serving shim for witness patrol |
| `collect_training_data.py` | Raw session data collector |
| `curate.py` | Session quality filtering |
| `extract_decisions.py` | Decision pair extraction |
| `synthetic_scenarios.py` | Hand-crafted scenario generator |
| `preprocess.py` | Multi-format chat preprocessor |
| `snapshot_format.py` | Rich snapshot formatting |
| `witness_tools.json` | Tool schema (374 lines) |
| `EXPERIMENT_REPORT.md` | Results from original experiments |
