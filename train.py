#!/usr/bin/env python3
"""
Fine-tuning script for witness tool-calling models.

Supports full fine-tune and LoRA on any HuggingFace causal LM.
Designed for CPU-only training.

Usage:
    python train.py --model SmolLM2-135M --format b --epochs 3
"""

import argparse
import json
import os
import time
import torch
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType


# Model registry
MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}


def load_dataset_jsonl(path: str) -> list:
    """Load a JSONL file where each line is a conversation (list of messages)."""
    convs = []
    with open(path) as f:
        for line in f:
            convs.append(json.loads(line))
    return convs


def format_conversation(messages: list, tokenizer) -> str:
    """Format a conversation using the tokenizer's chat template, or a simple fallback."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback: simple role-based formatting
        parts = []
        for m in messages:
            role = m["role"]
            content = m.get("content", "") or ""
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
        return "".join(parts)


def prepare_dataset(convs: list, tokenizer, max_length: int = 2048) -> Dataset:
    """Convert conversations to tokenized dataset."""
    texts = []
    skipped = 0
    for conv in convs:
        text = format_conversation(conv, tokenizer)
        # Rough token count check
        if len(text) / 4 > max_length * 1.5:
            skipped += 1
            continue
        texts.append(text)

    if skipped:
        print(f"  Skipped {skipped} conversations exceeding ~{max_length} tokens")

    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a witness model")
    parser.add_argument("--model", type=str, default="smollm2-135m",
                        choices=list(MODELS.keys()),
                        help="Model to fine-tune")
    parser.add_argument("--format", type=str, default="b",
                        choices=["a", "b", "c"],
                        help="Output format (a=inline tags, b=pure JSON, c=control tokens)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length")
    parser.add_argument("--lora", action="store_true",
                        help="Use LoRA (default: full fine-tune)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--dataset-dir", type=str, default="./dataset")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--use-chunked", action="store_true",
                        help="Use 2K-chunked dataset variant")
    parser.add_argument("--max-train", type=int, default=0,
                        help="Max training examples (0=all, useful for quick smoke tests)")
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args()

    model_id = MODELS[args.model]
    fmt_dir = f"format_{args.format}"
    if args.use_chunked:
        train_path = os.path.join(args.dataset_dir, fmt_dir, "chunked_2k", "train.jsonl")
        eval_path = os.path.join(args.dataset_dir, fmt_dir, "chunked_2k", "eval.jsonl")
    else:
        train_path = os.path.join(args.dataset_dir, fmt_dir, "train.jsonl")
        eval_path = os.path.join(args.dataset_dir, fmt_dir, "eval.jsonl")

    run_name = f"{args.model}_fmt{args.format}_{'lora' if args.lora else 'full'}_ep{args.epochs}"
    output_dir = os.path.join(args.output_dir, run_name)

    print(f"=" * 60)
    print(f"Training: {run_name}")
    print(f"Model:    {model_id}")
    print(f"Format:   {args.format}")
    print(f"Method:   {'LoRA r=' + str(args.lora_r) if args.lora else 'Full fine-tune'}")
    print(f"Epochs:   {args.epochs}")
    print(f"LR:       {args.lr}")
    print(f"Data:     {train_path}")
    print(f"Output:   {output_dir}")
    print(f"=" * 60)

    # Load tokenizer and model
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    # Apply LoRA if requested
    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA: {trainable/1e6:.1f}M trainable params ({trainable/n_params*100:.1f}%)")
    else:
        trainable = n_params
        print(f"Full fine-tune: {trainable/1e6:.1f}M trainable params")

    # Load and prepare dataset
    print("\nLoading dataset...")
    train_convs = load_dataset_jsonl(train_path)
    eval_convs = load_dataset_jsonl(eval_path)

    if args.max_train:
        train_convs = train_convs[:args.max_train]
        eval_convs = eval_convs[:min(args.max_train // 4, len(eval_convs))]

    print(f"  Train: {len(train_convs)} conversations")
    print(f"  Eval:  {len(eval_convs)} conversations")

    train_dataset = prepare_dataset(train_convs, tokenizer, args.max_length)
    eval_dataset = prepare_dataset(eval_convs, tokenizer, args.max_length)

    print(f"  Train dataset: {len(train_dataset)} examples")
    print(f"  Eval dataset:  {len(eval_dataset)} examples")

    # Sample a formatted example
    sample_text = train_dataset[0]["text"]
    print(f"\nSample (first 500 chars):\n{sample_text[:500]}")

    # Training config
    os.makedirs(output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        max_length=args.max_length,
        dataset_text_field="text",
        packing=False,
    )

    # Train
    print("\nStarting training...")
    start_time = time.time()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"Training complete: {run_name}")
    print(f"Wall time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Train loss: {result.training_loss:.4f}")
    print(f"{'=' * 60}")

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    # Quick inference test
    print("\nInference test:")
    model.eval()
    test_prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a Witness agent. You respond ONLY with JSON tool calls.\nFor each turn, output exactly one JSON object:\n{\"tool\": \"<tool_name>\", \"args\": {<arguments>}}"},
        {"role": "user", "content": "Polecats\n\n  ● gastown/furiosa  working\n    hq-wisp-2v214\n  ○ gastown/nux  done\n    gt-kvo.6"},
    ], tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(test_prompt, return_tensors="pt")
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    inf_time = time.perf_counter() - start

    generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Latency: {inf_time*1000:.0f}ms")
    print(f"  Output:  {generated[:300]}")

    # Save run metadata
    meta = {
        "run_name": run_name,
        "model": model_id,
        "format": args.format,
        "method": "lora" if args.lora else "full",
        "epochs": args.epochs,
        "lr": args.lr,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "train_loss": result.training_loss,
        "wall_time_seconds": elapsed,
        "n_params_total": n_params,
        "n_params_trainable": trainable,
        "inference_latency_ms": inf_time * 1000,
    }
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
