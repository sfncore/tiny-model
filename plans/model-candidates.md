# Tiny Model Candidates

Goal: Find the best sub-1B model for embedded tool calling (context-mode MCP, witness patrol, Gas Town agents).

## Evaluation Criteria

- Context window (must handle tool schemas + conversation)
- Native tool/function calling ability
- GGUF availability (for llama.cpp WebUI)
- VRAM footprint (RTX 3060, 12GB)
- Fine-tunability for our use cases

## Candidates

### 1. SmolLM2-135M (current)
- **HF**: `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Params**: 135M
- **Context**: 8,192 tokens
- **Architecture**: Llama
- **Tool calling**: Trained ours — 100% valid JSON, 69% correct tool (witness patrol)
- **GGUF**: Yes (official from bartowski)
- **VRAM**: ~300MB (bf16)
- **Status**: TRAINED, SERVING
- **Verdict**: Too small for general tool calling. 8K context insufficient for omp prompts. Good proof-of-concept for witness patrol only.

### 2. Granite 4.0-350M
- **HF**: `ibm-granite/granite-4.0-tiny-preview` (check exact ID)
- **Params**: 350M
- **Context**: 128,000 tokens
- **Architecture**: Hybrid Mamba/Transformer (constant-memory inference)
- **Tool calling**: Native, IBM-designed for agentic AI at the edge
- **GGUF**: Yes
- **VRAM**: ~500MB
- **Status**: NOT TESTED
- **Verdict**: STRONG CANDIDATE. 128K context solves the prompt size problem entirely. Native tool calling. Smallest model with real agent capability. Hybrid architecture means efficient inference.

### 3. Qwen2.5-Coder-0.5B
- **HF**: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- **Params**: 500M
- **Context**: 32,768 tokens
- **Architecture**: Qwen2 (Transformer)
- **Tool calling**: Code-focused, not native tool calling
- **GGUF**: Yes
- **VRAM**: ~600MB
- **Status**: NOT TESTED
- **Verdict**: Best for pure code generation. 32K context is good. Less suited for structured tool calling than Granite or Qwen3.

### 4. Qwen3-0.6B
- **HF**: `Qwen/Qwen3-0.6B`
- **Params**: 600M
- **Context**: 32,768 tokens
- **Architecture**: Qwen3 (Transformer)
- **Tool calling**: Qwen tool-call templates, thinking/non-thinking modes
- **GGUF**: Yes (347MB Q4)
- **VRAM**: ~700MB
- **Status**: Already in train.py model registry
- **Verdict**: STRONG CANDIDATE. Safe bet — already integrated in our training pipeline. 32K context handles omp prompts. Good tool-call template support.

### 5. Qwen3.5-0.8B
- **HF**: `Qwen/Qwen3.5-0.8B` (check exact ID)
- **Params**: 800M
- **Context**: 262,144 tokens
- **Architecture**: Gated DeltaNet
- **Tool calling**: Supported but less reliable at this scale
- **GGUF**: Yes
- **VRAM**: ~900MB
- **Status**: NOT TESTED
- **Verdict**: Massive context (262K) but newest/least proven. Thinking mode can loop at this scale. Worth testing if context is the priority.

### 6. TinyAgent-1.1B
- **HF**: `squeeze-ai-lab/TinyAgent-1.1B`
- **Params**: 1.1B
- **Context**: 2,048 tokens
- **Architecture**: TinyLlama (Llama)
- **Tool calling**: 80% success (beat GPT-4-Turbo on Berkeley benchmark)
- **GGUF**: Yes (668MB Q4)
- **VRAM**: ~1GB
- **Status**: NOT TESTED
- **Verdict**: POOR FIT. Impressive benchmark but 2K context is a dealbreaker. Trained specifically for MacOS assistant tasks (email/calendar/contacts). Needs separate ToolRAG model. Not general-purpose.

## Recommendation

**Primary**: Granite 4.0-350M — smallest model with real agent capability, 128K context
**Fallback**: Qwen3-0.6B — proven, already in pipeline, 32K context
**Stretch**: Qwen3.5-0.8B — if 262K context matters

## Next Steps

1. Download Granite 4.0-350M GGUF and test via llama-server WebUI
2. Test native tool calling with context-mode MCP tool schemas
3. If promising, fine-tune on witness patrol + context-mode synthetic data
4. Compare against Qwen3-0.6B on same tasks
