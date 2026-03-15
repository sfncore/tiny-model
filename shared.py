"""Shared utilities for the tiny_model project."""

import glob
import logging
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

log = logging.getLogger(__name__)


def load_model(checkpoint_path: str):
    """Load model and tokenizer from checkpoint.

    Returns (model, tokenizer) tuple. Model is in eval mode with float32 dtype.
    """
    log.info("Loading model from %s", checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Loaded: %.1fM params", n_params / 1e6)
    return model, tokenizer


def find_session_files(session_dirs: list, min_size: int = 5000,
                       max_size: int = 2_000_000) -> list:
    """Find all witness session JSONL files within size bounds.

    Skips missing directories and unreadable files gracefully.
    """
    files = []
    for d in session_dirs:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "*.jsonl")):
            try:
                size = os.path.getsize(path)
                if min_size <= size <= max_size:
                    files.append(path)
            except OSError:
                continue
    return sorted(files)
