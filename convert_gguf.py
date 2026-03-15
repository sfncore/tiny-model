#!/usr/bin/env python3
"""
Convert a trained checkpoint to GGUF format for llama.cpp serving.

Usage:
    python convert_gguf.py --checkpoint ./checkpoints/granite-350m_fmtb_full_ep3/final
    python convert_gguf.py --checkpoint ./checkpoints/granite-350m_fmtb_full_ep3/final --outtype q8_0
    python convert_gguf.py --checkpoint ./checkpoints/granite-350m_fmtb_full_ep3/final --outdir ./models
"""

import argparse
import subprocess
import sys
from pathlib import Path

LLAMA_CPP_CONVERTER = "/tmp/llama.cpp/convert_hf_to_gguf.py"


def main():
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to GGUF")
    parser.add_argument("--checkpoint", required=True, help="Path to HF checkpoint directory")
    parser.add_argument("--outdir", default="./models", help="Output directory for GGUF file")
    parser.add_argument("--outtype", default="f16", choices=["f16", "f32", "bf16", "q8_0"],
                        help="Output precision (default: f16)")
    parser.add_argument("--converter", default=LLAMA_CPP_CONVERTER,
                        help="Path to convert_hf_to_gguf.py from llama.cpp")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    converter = Path(args.converter)
    if not converter.exists():
        print(f"ERROR: Converter not found: {converter}")
        print("Clone llama.cpp first: git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp")
        print("Then install gguf: pip install /tmp/llama.cpp/gguf-py/")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Derive output name from checkpoint path
    # e.g. checkpoints/granite-350m_fmtb_full_ep3/final -> granite-350m-witness.gguf
    parts = checkpoint.parts
    for p in parts:
        if "checkpoint" not in p.lower() and "final" not in p.lower():
            name = p
    name = name.replace("_fmtb_full_ep3", "").replace("_", "-")
    outfile = outdir / f"{name}-witness.gguf"

    print(f"Converting: {checkpoint}")
    print(f"Output:     {outfile}")
    print(f"Type:       {args.outtype}")

    cmd = [
        sys.executable, str(converter),
        str(checkpoint),
        "--outfile", str(outfile),
        "--outtype", args.outtype,
    ]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Conversion failed (exit {result.returncode})")
        sys.exit(1)

    size_mb = outfile.stat().st_size / (1024 * 1024)
    print(f"\nDone: {outfile} ({size_mb:.0f}MB)")


if __name__ == "__main__":
    main()
