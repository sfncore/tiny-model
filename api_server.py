#!/usr/bin/env python3
"""
Lightweight OpenAI-compatible API server for SmolLM2 models.
Serves both trained (witness) and untrained (base) models.
Supports streaming (SSE) and non-streaming responses.

Usage:
    python api_server.py                          # default: both models on port 8080
    python api_server.py --port 8080
    python api_server.py --base-only
    python api_server.py --witness-only
"""

import argparse
import json
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

MODELS = {}

BASE_MODELS = {
    "smollm2-base": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "granite-base": "ibm-granite/granite-4.0-350m",
}
WITNESS_CHECKPOINT = "./checkpoints/smollm2-135m_fmtb_full_ep3/final"


def load_model(name: str, path: str, device: str):
    print(f"Loading {name} from {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {n_params/1e6:.1f}M params on {device}")
    MODELS[name] = {"model": model, "tokenizer": tokenizer}


def normalize_content(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(
            p.get("text", "") for p in c
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(c)


def resolve_model(name: str) -> str:
    if name in MODELS:
        return name
    for k in MODELS:
        if name in k or k in name:
            return k
    return list(MODELS.keys())[0]


def run_inference(model_name: str, messages: list, max_tokens: int, temperature: float):
    entry = MODELS[model_name]
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    device = next(model.parameters()).device

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.perf_counter() - start
    generated = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    completion_tokens = len(out[0]) - input_len

    return generated, input_len, completion_tokens, latency


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": name, "object": "model"} for name in MODELS],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    req_model = body.get("model", "smollm2-witness")
    req_messages = body.get("messages", [])
    req_max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 512
    req_temperature = body.get("temperature", 0.0)
    req_stream = body.get("stream", False)

    model_name = resolve_model(req_model)
    messages = [
        {"role": m.get("role", "user"), "content": normalize_content(m.get("content"))}
        for m in req_messages
    ]

    generated, input_len, completion_tokens, latency = run_inference(
        model_name, messages, req_max_tokens, req_temperature
    )

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req_stream:
        def stream_response():
            # Single chunk with full content (model is too small for real token streaming)
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": generated},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk with finish_reason
            done_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            if body.get("stream_options", {}).get("include_usage"):
                done_chunk["usage"] = {
                    "prompt_tokens": input_len,
                    "completion_tokens": completion_tokens,
                    "total_tokens": input_len + completion_tokens,
                }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": completion_tokens,
            "total_tokens": input_len + completion_tokens,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated model names to load (e.g. 'granite-base,smollm2-base') or 'all'")
    parser.add_argument("--witness-checkpoint", type=str, default=WITNESS_CHECKPOINT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return

    device = "cuda"
    requested = args.models.split(",") if args.models != "all" else list(BASE_MODELS.keys()) + ["smollm2-witness"]

    for name in requested:
        name = name.strip()
        if name in BASE_MODELS:
            load_model(name, BASE_MODELS[name], device)
        elif name == "smollm2-witness":
            load_model("smollm2-witness", args.witness_checkpoint, device)
        else:
            print(f"Unknown model: {name}. Available: {list(BASE_MODELS.keys()) + ['smollm2-witness']}")

    print(f"\nServing {list(MODELS.keys())} on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
