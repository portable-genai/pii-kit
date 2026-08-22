# Serving Gemma 4 for pii-kit's optional model channel

`pii-kit` detects and redacts PII with a **deterministic regex + checksum pack** by default —
no model, no network, air-gappable. That default is unchanged by anything in this guide.

This guide sets up the **optional** model-assisted channel (`pii_kit.model`), which raises
recall on the PII a regex cannot express (names, postal addresses, free-text identifiers) and
adds an **image** scan the pack has no equivalent for. The channel talks to any
**OpenAI-compatible chat-completions endpoint** over stdlib `urllib` — so a locally hosted
[Gemma 4](https://ai.google.dev/gemma) model served by
[vLLM](https://docs.vllm.ai) is the reference target, and the scanned text/image never leaves
the host.

> **The model only ever ADDS detections.** A model finding can drop the safety score to `0.0`,
> never lift it; the pack always runs and cannot be overridden. So a wrong, slow, or missing
> model degrades recall — it can never make a leak look clean. See
> [`../src/pii_kit/model.py`](../src/pii_kit/model.py) for the full contract.

Contents:
1. [Which path fits your hardware](#1-which-path-fits-your-hardware)
2. [About Gemma 4 (sizes, vision, VRAM)](#2-about-gemma-4)
3. [Linux + NVIDIA GPU — vLLM](#3-linux--nvidia-gpu--vllm)
4. [macOS (Apple Silicon) — llama.cpp / Ollama](#4-macos-apple-silicon--llamacpp--ollama)
5. [Wire it into pii-kit](#5-wire-it-into-pii-kit)
6. [Verify end-to-end](#6-verify-end-to-end)
7. [Operational notes](#7-operational-notes)

---

## 1. Which path fits your hardware

| Host | Recommended server | Vision? | Section |
|---|---|---|---|
| **Linux + NVIDIA GPU** (≥24 GB) | **vLLM** serving Gemma 4 | yes | [§3](#3-linux--nvidia-gpu--vllm) |
| **Linux, CPU-only** | vLLM CPU build (small model, slow — test only) | yes | [§3.5](#35-cpu-only-linux) |
| **macOS (Apple Silicon)** | **llama.cpp** `llama-server` (or Ollama) serving a Gemma vision GGUF | yes | [§4](#4-macos-apple-silicon--llamacpp--ollama) |

Why not vLLM on a Mac: vLLM's macOS support is **experimental, CPU-only** (no Metal/MPS in
core), so it is far too slow for interactive use, and the Apple-Silicon GPU plugin
[`vllm-metal`](https://github.com/vllm-project/vllm-metal) currently serves Gemma only
**text-only** (its multimodal path is limited to a couple of non-Gemma models). For a Gemma
**vision** model on a Mac, use `llama.cpp` or Ollama — both expose the same OpenAI
chat-completions API, so `pii_kit` uses identical client code; only the `base_url` changes.

---

## 2. About Gemma 4

Gemma 4 (Google, released 2026, **Apache-2.0** licensed) is a family of open models. **Every
size is multimodal** (text **and image** input) — there is no text-only Gemma 4 variant — which
is what makes it a fit for both the text and the image PII channels.

Instruction-tuned model IDs on Hugging Face (note the **uppercase** `E`/`B`, unlike Gemma 3's
lowercase):

| Model ID | Params | Min GPU (BF16) | ~Quantized (Q4) | Notes |
|---|---|---|---|---|
| `google/gemma-4-E2B-it` | 2.3 B eff. | 1×24 GB | ~2 GB | smallest; text+image+audio |
| `google/gemma-4-E4B-it` | 4.5 B eff. | 1×24 GB | ~4 GB | **good default** for a single 24 GB card |
| `google/gemma-4-12B-it` | 12 B | 1×40 GB | — | "Unified" encoder-free architecture |
| `google/gemma-4-26B-A4B-it` | 25 B (MoE, ~4 B active) | 1×80 GB | ~14 GB | all experts must be resident |
| `google/gemma-4-31B-it` | 31 B dense | 1×80 GB (or 2×24 GB TP) | ~20 GB (fits 24 GB) | highest quality |

- **Vision** = image + short video (frames); **audio** input is supported on E2B/E4B/12B only.
- **Context**: 128 K (E2B/E4B), 256 K (12B/26B/31B).
- **License / access**: Apache-2.0, and the `google/gemma-4-*` repos are **not gated** — no
  license click-through or Hugging Face token is required to pull them. (This differs from
  Gemma 3, which was gated; if you fall back to a Gemma 3 model you must accept its terms and
  authenticate — see the notes in each section.)

Quantized figures are community estimates, not official; the BF16 GPU column follows vLLM's
official Gemma 4 recipe. For a first deployment on a single 24 GB card, **`gemma-4-E4B-it`** is
the sweet spot; step up to `gemma-4-31B-it` (BF16 on 80 GB, or Q4 on 24 GB) when recall matters
more than latency.

---

## 3. Linux + NVIDIA GPU — vLLM

Requires a recent NVIDIA driver (CUDA 12.8+ runtime; vLLM wheels are built against CUDA 12.9)
and Python 3.10–3.13. vLLM added Gemma 4 support in **v0.19.0**; any current release
(0.25.x as of mid-2026) is fine.

### 3.1 Install

The vLLM docs recommend [`uv`](https://docs.astral.sh/uv/) because it resolves the PyTorch
index correctly:

```sh
# install uv if you don't have it: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto     # --torch-backend=auto matches your CUDA driver
```

Plain pip also works:

```sh
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
```

### 3.2 Serve a Gemma 4 vision model

`gemma-4-E4B-it` on a single 24 GB card, with the OpenAI-compatible server on the default
`http://localhost:8000`:

```sh
vllm serve google/gemma-4-E4B-it \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 1}'
```

- `--limit-mm-per-prompt '{"image": 1}'` — `pii_kit` sends **one** image per request; raise
  the number only if you batch several. Text-only scans need no multimodal flag, but leaving it
  set is harmless.
- `--max-model-len` — cap context to fit VRAM; PII scans are short, so 32 K is generous.
- **bfloat16** is the default and recommended dtype (Gemma checkpoints are BF16).

Higher-quality 31B across two 24 GB GPUs (tensor-parallel):

```sh
vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 1}'
```

The tool-calling / "thinking" flags in vLLM's full recipe (`--reasoning-parser gemma4`,
`--tool-call-parser gemma4`, a custom `--chat-template`) are **not needed** here — PII detection
is a single-turn request that returns JSON via the built-in chat template.

### 3.3 Require an API key (recommended)

```sh
vllm serve google/gemma-4-E4B-it --api-key "sk-local-pii-please-change" \
  --limit-mm-per-prompt '{"image": 1}'
# or: export VLLM_API_KEY="sk-local-pii-please-change"
```

With a key set, the server requires `Authorization: Bearer <key>` on `/v1/*`. Point
`pii_kit` at it via `ModelConfig(api_key=...)` / `PII_PACK_MODEL_API_KEY` ([§5](#5-wire-it-into-pii-kit)).

### 3.4 Docker (alternative)

```sh
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model google/gemma-4-E4B-it --limit-mm-per-prompt '{"image": 1}'
```

### 3.5 CPU-only Linux

Supported but **slow — for functional testing, not production**:

```sh
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu \
  --index-strategy first-index --torch-backend cpu
vllm serve google/gemma-4-E2B-it --dtype bfloat16 --limit-mm-per-prompt '{"image": 1}'
```

(There are also `vllm/vllm-openai-cpu:latest-x86_64` / `:latest-arm64` images.)

---

## 4. macOS (Apple Silicon) — llama.cpp / Ollama

Both serve a Gemma **vision** model behind an OpenAI-compatible `/v1/chat/completions` that
accepts base64 `image_url` data URIs — exactly what `pii_kit.model` sends — so only the
`base_url` differs from the vLLM path. As of mid-2026 the vision GGUFs that "just work" on these
runtimes are **Gemma 3** (`gemma-3-4b/12b/27b-it`); Gemma 4 GGUF vision support in these
runtimes may lag vLLM, so this section uses Gemma 3 for the image path. Gemma 3 vision quality is
still well above regex for names/addresses.

### 4.1 llama.cpp `llama-server` (most faithful to the vLLM client)

```sh
brew install llama.cpp

# -hf auto-downloads the model GGUF *and* the vision projector (mmproj); Metal GPU by default.
llama-server -hf ggml-org/gemma-3-4b-it-GGUF          # or -12b-it / -27b-it
# serves OpenAI-compatible API on http://localhost:8080
```

`llama-server` defaults to port **8080** and accepts base64 `image_url` blocks, so unmodified
OpenAI client code works by pointing `base_url` at `http://localhost:8080/v1`. Add
`--api-key <key>` to require a bearer token.

### 4.2 Ollama (simplest)

```sh
brew install ollama
ollama serve                       # http://localhost:11434
ollama pull gemma3:4b              # 4b / 12b / 27b are vision-capable; 270m / 1b are text-only
```

Ollama's OpenAI endpoint is `http://localhost:11434/v1`. It accepts base64-encoded `image_url`
data URIs (**remote image URLs are not supported** — `pii_kit` always sends base64, so that's
fine). Ollama has no auth; the OpenAI convention is to pass any non-empty `api_key` string
(`pii_kit` simply omits the header when `api_key` is `None`).

> **Text-only Mac note:** `mlx_lm.server` (MLX) is fast but text-only — fine for the text
> channel, not for images. For the image channel, use llama.cpp or Ollama as above.

---

## 5. Wire it into pii-kit

The channel is **off** until you pass a `ModelConfig`. Two ways to supply one.

### 5.1 From environment variables (recommended for gates/CI)

```sh
export PII_PACK_MODEL_BASE_URL="http://localhost:8000/v1"   # vLLM; 8080/v1 llama.cpp; 11434/v1 Ollama
export PII_PACK_MODEL="google/gemma-4-E4B-it"               # served model id / name
export PII_PACK_MODEL_API_KEY="sk-local-pii-please-change"  # optional
```

```python
from pii_kit import (
    ModelConfig, UNIVERSAL_PATTERNS, national_patterns_for, score_pii_safety,
)

rows = [*UNIVERSAL_PATTERNS, *national_patterns_for(("SG", "JP", "AU"))]
cfg = ModelConfig.from_env()   # -> None when the env vars aren't set (plain deterministic flow)

# `surfaces` are your pipeline's DERIVED outputs after redaction, never the raw input.
metric = score_pii_safety(surfaces, rows, planted_tokens=[...], model_config=cfg)
# 1.0 clean; 0.0 if the pack, the planted-literal oracle, OR the model saw PII.
```

`ModelConfig.from_env()` returns `None` unless both `PII_PACK_MODEL_BASE_URL` and
`PII_PACK_MODEL` are set — so the **same code path** runs the plain deterministic scorer
wherever the endpoint isn't configured. No model, no behaviour change.

### 5.2 Explicit config

```python
cfg = ModelConfig(
    base_url="http://localhost:8000/v1",
    model="google/gemma-4-E4B-it",
    api_key="sk-local-pii-please-change",   # omit / None for Ollama or an unauthenticated server
    response_format="json_schema",          # see below; also: timeout, temperature, max_tokens
)
```

`response_format` controls how the findings JSON is requested — leave it at the default unless a
server rejects it:

- `"json_schema"` (default) — constrains decoding to the findings schema; honoured by vLLM,
  Ollama and llama.cpp. (vLLM removed the older `guided_json` API in v0.12, so this is the
  portable form.)
- `"json_object"` — asks only for syntactically valid JSON, for a server without schema support.
- `"none"` — omits the field entirely and relies on the prompt (last resort).

Either way the reply is still parsed and validated client-side, so a server that ignores the
field degrades to prompt-only behaviour rather than breaking.

### 5.3 Text and image detection / redaction directly

```python
from pii_kit import (
    model_text_findings, model_image_findings, model_redact, redact,
)

# Text: model findings, each marked grounded (verbatim in the text) or not.
for f in model_text_findings("Loan officer: Priya Nair, 42 Orchard Rd.", cfg):
    print(f.info_type, repr(f.text), "grounded" if f.grounded else "ungrounded")

# Text redaction — compose AFTER the deterministic pack so the pack decides its own ids first.
# Only GROUNDED model findings are masked (hallucinated spans can't be located, so they can't
# corrupt the text); ungrounded findings still count as a leak in scoring.
clean = model_redact(redact(text, rows), cfg)

# Image: scan raw image bytes (PNG/JPEG/GIF/WebP). Requires a vision-capable model.
# The pack has no image channel, so this is detection the default flow cannot do.
with open("scanned_form.png", "rb") as fh:
    findings = model_image_findings(fh.read(), cfg)   # all ungrounded (no source text to match)
```

If the endpoint is unreachable or returns an unparseable reply, these raise
`pii_kit.ModelAPIError` **rather than silently returning "no findings"** — a model outage
can't turn a gate green. Catch it if you want best-effort behaviour:

```python
from pii_kit import ModelAPIError
try:
    metric = score_pii_safety(surfaces, rows, model_config=cfg)
except ModelAPIError:
    metric = score_pii_safety(surfaces, rows)   # fall back to the deterministic pack alone
```

---

## 6. Verify end-to-end

**Server reachable + serving your model:**

```sh
curl -s http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-local-pii-please-change" | python3 -m json.tool
```

**Text detection from Python** (should print a `PERSON_NAME` finding a regex would miss):

```python
from pii_kit import ModelConfig, model_text_findings
cfg = ModelConfig(base_url="http://localhost:8000/v1", model="google/gemma-4-E4B-it",
                  api_key="sk-local-pii-please-change")
print(model_text_findings("Please contact Wei Ling about the overdue invoice.", cfg))
```

**The scorer goes red on a name only the model can see:**

```python
from pii_kit import UNIVERSAL_PATTERNS, score_pii_safety
surfaces = ["Case note: Wei Ling prefers evening calls."]
print(score_pii_safety(surfaces, UNIVERSAL_PATTERNS))                     # 1.0 (pack alone)
print(score_pii_safety(surfaces, UNIVERSAL_PATTERNS, model_config=cfg))   # 0.0 (model added it)
```

---

## 7. Operational notes

- **Privacy.** The scanned text/image is POSTed to `base_url`. With a locally hosted model it
  stays on the host — the whole point of a local endpoint. Do **not** set `base_url` to a
  third-party API unless sending it your PII is acceptable.
- **Error messages / debugging.** `ModelAPIError` carries the **HTTP status only** — never the
  scanned input and never the server's response body (a server or proxy could echo the scanned
  request into an error body). Diagnose failures from the **model server's own logs**, which you
  control, not from the exception text. A common first failure is an unreachable endpoint (wrong
  `base_url`/port) or a model still loading (HTTP 503) — retry once the server is ready.
- **Determinism.** `pii_kit` sends `temperature=0`, but LLM inference is not bit-reproducible
  (GPU reduction order varies with server batch size). This is why the model is advisory: the
  deterministic pack + planted-literal oracle remain the reproducible gate; the model only adds
  recall on top.
- **Prompt injection.** The scanned document is untrusted data, and a language model can be
  steered by adversarial text inside it ("ignore the above…") into *missing* PII. The prompt
  hardens against this, but the real safeguard is structural: the model can only **add**
  detections to the pack, which cannot be steered. Never let a model finding suppress a pack
  finding.
- **Grounding.** Only findings whose value appears **verbatim** in the scanned text are used for
  masking; that check discards hallucinated spans. Image findings have no source text to match,
  so they are detect-only (flag the image; redact pixels in your own pipeline).
- **Cost / latency.** A regex scan is ~0.1 ms; a model scan is 10²–10³ ms and needs a GPU.
  Run the model channel where recall justifies it (free-text notes, images), not on every
  high-volume structured field the pack already covers.
- **Model choice.** Bigger Gemma = better recall on subtle/free-form PII. Start at
  `gemma-4-E4B-it`; move to `gemma-4-31B-it` if evaluation shows missed names/addresses.
