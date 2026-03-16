#!/usr/bin/env python3
"""
Lightweight OpenAI-compatible API server for tiny witness models.
Supports streaming (SSE) and non-streaming responses.

Usage:
    python api_server.py                        # uses serve.yaml
    python api_server.py --config custom.yaml   # custom config
    python api_server.py --model witness        # override: load only 'witness'
"""

import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

# Global state
MODELS = {}
CONFIG = {}
_inference_pool = ThreadPoolExecutor(max_workers=1)


def load_config(config_path: str) -> dict:
    """Load and validate serve.yaml config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Defaults
    cfg.setdefault("server", {})
    cfg["server"].setdefault("host", "127.0.0.1")
    cfg["server"].setdefault("port", 8081)

    cfg.setdefault("inference", {})
    cfg["inference"].setdefault("max_tokens_cap", 2048)
    cfg["inference"].setdefault("max_tokens_default", 512)
    cfg["inference"].setdefault("max_prompt_length", 8192)
    cfg["inference"].setdefault("timeout_seconds", 30)

    cfg.setdefault("models", {})
    return cfg


def load_model(name: str, path: str, device: str):
    """Load a model onto the specified device."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {name} from {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=True
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


def resolve_model(name: str) -> str | None:
    if not MODELS:
        return None
    if name in MODELS:
        return name
    for k in MODELS:
        if name in k or k in name:
            return k
    # Return default model (first one, or the one marked default in config)
    default = CONFIG.get("_default_model")
    if default and default in MODELS:
        return default
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

    timeout = CONFIG.get("inference", {}).get("timeout_seconds", 30)

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
    if latency > timeout:
        raise TimeoutError(f"Inference took {latency:.1f}s (limit: {timeout}s)")
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
    inf = CONFIG.get("inference", {})
    max_tokens_cap = inf.get("max_tokens_cap", 2048)
    max_tokens_default = inf.get("max_tokens_default", 512)
    max_prompt_length = inf.get("max_prompt_length", 8192)
    timeout = inf.get("timeout_seconds", 30)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={
            "error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}
        })

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={
            "error": {"message": "Request body must be a JSON object", "type": "invalid_request_error"}
        })

    if not MODELS:
        return JSONResponse(status_code=503, content={
            "error": {"message": "No models loaded.", "type": "server_error"}
        })

    req_messages = body.get("messages", [])
    if not isinstance(req_messages, list) or len(req_messages) == 0:
        return JSONResponse(status_code=400, content={
            "error": {"message": "'messages' must be a non-empty array", "type": "invalid_request_error"}
        })

    for i, m in enumerate(req_messages):
        if not isinstance(m, dict) or "role" not in m:
            return JSONResponse(status_code=400, content={
                "error": {"message": f"messages[{i}] must have a 'role' field", "type": "invalid_request_error"}
            })

    req_max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or max_tokens_default
    if not isinstance(req_max_tokens, int) or req_max_tokens < 1:
        return JSONResponse(status_code=400, content={
            "error": {"message": "'max_tokens' must be a positive integer", "type": "invalid_request_error"}
        })
    req_max_tokens = min(req_max_tokens, max_tokens_cap)

    req_temperature = body.get("temperature", 0.0)
    if not isinstance(req_temperature, (int, float)) or req_temperature < 0:
        return JSONResponse(status_code=400, content={
            "error": {"message": "'temperature' must be non-negative", "type": "invalid_request_error"}
        })

    req_stream = body.get("stream", False)

    req_model = body.get("model", CONFIG.get("_default_model", "witness"))
    model_name = resolve_model(req_model)
    if model_name is None:
        return JSONResponse(status_code=503, content={
            "error": {"message": "No models loaded", "type": "server_error"}
        })

    messages = [
        {"role": m.get("role", "user"), "content": normalize_content(m.get("content"))}
        for m in req_messages if isinstance(m, dict)
    ]

    if not messages:
        return JSONResponse(status_code=400, content={
            "error": {"message": "No valid messages", "type": "invalid_request_error"}
        })

    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars > max_prompt_length:
        return JSONResponse(status_code=400, content={
            "error": {"message": f"Prompt too long ({total_chars} > {max_prompt_length})", "type": "invalid_request_error"}
        })

    try:
        future = _inference_pool.submit(run_inference, model_name, messages, req_max_tokens, req_temperature)
        generated, input_len, completion_tokens, latency = future.result(timeout=timeout)
    except FuturesTimeoutError:
        future.cancel()
        return JSONResponse(status_code=504, content={
            "error": {"message": f"Inference timed out ({timeout}s)", "type": "server_error"}
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": {"message": f"Inference error: {type(e).__name__}: {e}", "type": "server_error"}
        })

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req_stream:
        def stream_response():
            chunk = {
                "id": chat_id, "object": "chat.completion.chunk",
                "created": created, "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": generated}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            done_chunk = {
                "id": chat_id, "object": "chat.completion.chunk",
                "created": created, "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            if body.get("stream_options", {}).get("include_usage"):
                done_chunk["usage"] = {"prompt_tokens": input_len, "completion_tokens": completion_tokens, "total_tokens": input_len + completion_tokens}
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_response(), media_type="text/event-stream")

    return {
        "id": chat_id, "object": "chat.completion", "created": created, "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": generated}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_len, "completion_tokens": completion_tokens, "total_tokens": input_len + completion_tokens},
    }


def main():
    parser = argparse.ArgumentParser(description="Tiny model API server")
    parser.add_argument("--config", type=str, default="serve.yaml", help="Path to serve.yaml config")
    parser.add_argument("--model", type=str, default=None, help="Override: load only this model name from config")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    parser.add_argument("--host", type=str, default=None, help="Override server host")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return

    global CONFIG
    CONFIG = load_config(args.config)

    host = args.host or CONFIG["server"]["host"]
    port = args.port or CONFIG["server"]["port"]
    device = "cuda"

    # Load models from config
    models_cfg = CONFIG.get("models", {})
    for name, mcfg in models_cfg.items():
        if not isinstance(mcfg, dict):
            continue
        if args.model and name != args.model:
            continue
        if not mcfg.get("enabled", True):
            continue
        load_model(name, mcfg["path"], device)
        if mcfg.get("default"):
            CONFIG["_default_model"] = name

    if not MODELS:
        print("ERROR: No models loaded. Check serve.yaml.")
        return

    print(f"\nServing {list(MODELS.keys())} on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
