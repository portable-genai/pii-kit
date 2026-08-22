# pii-kit

Jurisdiction-aware **PII pattern pack** for detection and redaction. One source of truth for
what a national identifier looks like, the checksum validators that harden those shapes,
RE2-safe forms for matchers with no lookaround, and a two-part redaction-safety scorer that
cannot go falsely green.

**Pure standard library. Zero runtime dependencies.** It installs and runs on an air-gapped
host.

An **optional model-assisted channel** ([`pii_kit.model`](src/pii_kit/model.py)) can raise
recall on the free-text PII a regex cannot express (names, addresses) and scan **images**, by
calling a locally hosted OpenAI-compatible endpoint (e.g. Gemma 4 on vLLM — see
[docs/vllm-gemma-setup.md](docs/vllm-gemma-setup.md)). It is **off by default**, advisory (it
only ever *adds* detections, never suppresses one or turns a leak green), and itself stdlib-only
(`urllib`, no `openai` package). The deterministic pack is always the authority.

## What you get

```python
from pii_kit import (
    NATIONAL_ID_PATTERNS,   # {"SG": [...], "HK": [...], "JP": [...], "AU": [...], "IN", "GB"}
    UNIVERSAL_PATTERNS,     # (EMAIL, PHONE_INTL)
    national_patterns_for,  # de-duplicated national rows for a set of jurisdictions
    re2_pattern_for,        # RE2-safe source for a row (e.g. Google Cloud DLP custom info types)
    validators,             # sg_nric_valid, hk_hkid_valid, jp_my_number_valid, au_tfn_valid, ...
    pack_leak,              # scorer half 1: scan with the same rows the redactor uses
    planted_leak,           # scorer half 2: pack-independent literal oracle
    score_pii_safety,       # combine both (+ optional model) into a 0.0 / 1.0 safety metric
    redact,                 # optional shared masking convenience

    # optional model-assisted channel (off unless a ModelConfig is supplied)
    ModelConfig,            # endpoint details; ModelConfig.from_env() reads PII_PACK_MODEL_*
    model_leak,             # scorer half 3 (text): True if the model reports any PII
    model_image_leak,       # scorer half 3 (image): True if the model reports any PII
    model_text_findings,    # model PII findings in text (each marked grounded / ungrounded)
    model_image_findings,   # model PII findings in an image (vision model; detect-only)
    model_redact,           # mask the model's grounded findings; compose after redact()
    ModelAPIError,          # raised on endpoint failure — never degrades to a silent pass
)
```

Each row is a 3-tuple `(info_type, re.Pattern, validator | None)`. A `validator`, when present,
is a `str -> bool` predicate applied to each raw match, so a detection counts only when the
value is a genuine identifier (the checksum passes), which keeps false positives low with no ML.

### Coverage

| Jurisdiction | Identifiers |
|---|---|
| Singapore (SG) | NRIC / FIN (case-insensitive), local phone |
| Hong Kong (HK) | HKID (parenthesised on shape; bare keyed form checksum-gated) |
| Japan (JP) | My Number (checksum-gated; grouped `1234 5678 9018` form included) |
| Australia (AU) | TFN (checksum-gated) |
| India (IN), UK (GB) | PAN, Aadhaar, NINO (reference rows) |
| Universal | Email, international phone |

Standalone checksum validators are also provided for IBAN, Luhn (card PAN), AU ABN and AU
Medicare.

## Install

```sh
pip install "pii-kit @ git+https://github.com/portable-genai/pii-kit@v0.0.1"
```

or clone and `pip install -e .`.

## Quick start

### Redact

```python
from pii_kit import UNIVERSAL_PATTERNS, national_patterns_for, redact

rows = [*UNIVERSAL_PATTERNS, *national_patterns_for(("SG", "JP", "AU"))]
clean = redact("Applicant S1234567D, My Number 1234 5678 9018, reach a@b.com", rows)
# -> "Applicant [REDACTED:SG_NRIC_FIN], My Number [REDACTED:JP_MY_NUMBER], reach [REDACTED:EMAIL_ADDRESS]"
```

### Score redaction safety in a test or gate

```python
from pii_kit import national_patterns_for, UNIVERSAL_PATTERNS, score_pii_safety

rows = [*UNIVERSAL_PATTERNS, *national_patterns_for(("SG",))]
planted = "S1234567D"
# `surfaces` are the DERIVED outputs of your pipeline (logs, records, model inputs) after
# redaction, never an echo of the raw input.
metric = score_pii_safety(surfaces, rows, planted_tokens=[planted])   # 1.0 clean, 0.0 leak
```

## Model-assisted detection (optional)

The regex + checksum pack is exact on **structured** identifiers (national IDs, email, phone,
card) and needs no model. It cannot express **free-text** PII — a person's name, a postal
address — and it cannot see **images** at all. A locally hosted language model closes that gap:
it is a **third detection channel** alongside `pack_leak` and `planted_leak`, never a
replacement for either. Point `pii_kit` at any OpenAI-compatible endpoint (vLLM, Ollama,
`llama.cpp`); the text/image stays on the host. Full setup, including **Gemma 4 on vLLM for
Linux and macOS**, is in [docs/vllm-gemma-setup.md](docs/vllm-gemma-setup.md).

```python
from pii_kit import ModelConfig, UNIVERSAL_PATTERNS, national_patterns_for, score_pii_safety

rows = [*UNIVERSAL_PATTERNS, *national_patterns_for(("SG", "JP", "AU"))]
cfg = ModelConfig.from_env()   # PII_PACK_MODEL_BASE_URL / PII_PACK_MODEL / *_API_KEY; None if unset

# Same call, one extra kwarg. With cfg=None it is exactly the deterministic scorer above.
metric = score_pii_safety(surfaces, rows, planted_tokens=[planted], model_config=cfg)
```

Design guarantees (see [`src/pii_kit/model.py`](src/pii_kit/model.py)):

- **Advisory, red-only.** A model finding can drop the score to `0.0`, never raise it; the pack
  always runs and cannot be overridden. A wrong/slow/absent model degrades recall, never safety.
- **Grounded masking.** Only findings whose value is a **verbatim substring** of the text are
  masked (`model_redact`), which discards hallucinated spans; ungrounded findings still count as
  a leak. Masking is **span-based against the original text** (overlaps merged, labels
  deterministic), so an inserted mask token can't be corrupted by a later value. Image findings
  are inherently ungrounded — detect-only.
- **Fails loud.** An unreachable endpoint or unparseable reply raises `ModelAPIError` instead of
  degrading to "no findings" (which would score a vacuous pass). Its message carries the **HTTP
  status only** — never the scanned input, and never the server's response body.
- **Opt-in and stdlib-only.** No `ModelConfig` ⇒ no HTTP call. `urllib` only — the `openai`
  package is never required, so the zero-dependency promise holds.

Beyond the scorer, the channel exposes direct helpers: `model_leak` / `model_image_leak` return
a boolean, `model_text_findings` / `model_image_findings` return typed `ModelFinding`s (each with
a `grounded` flag), and `model_redact` masks the grounded text findings. Image scans (need a
vision-capable model such as Gemma 4):

```python
from pii_kit import ModelConfig, model_image_findings

cfg = ModelConfig(base_url="http://localhost:8000/v1", model="google/gemma-4-E4B-it")
with open("scanned_form.png", "rb") as fh:
    findings = model_image_findings(fh.read(), cfg)   # PII read out of the image
```

## Composing rows: order is yours

The pack exposes rows but does **not** bake in their application order, because the right order
is application-specific:

```python
from pii_kit import UNIVERSAL_PATTERNS, national_patterns_for

national = national_patterns_for(("SG", "HK", "JP", "AU"))

# A bare-digit-catch-all account row (\b\d{9,17}\b) subsumes the contiguous national-id shapes,
# so it must run LAST or it masks a My Number as an account number.
rows = [*UNIVERSAL_PATTERNS, *national, account_row]

# A specific-shape account row (\b\d{3}-\d{6}-\d\b) is instead bitten into by the AU TFN row
# (the leading 9 digits can pass the TFN checksum), so it must run FIRST.
rows = [*UNIVERSAL_PATTERNS, account_row, *national]
```

The account-number row itself is application-specific PII with an application-specific shape, so
you supply it; it is not in the pack.

## Why the two-part scorer

A leak check scored only off the same rows the redactor masks with is a closed loop: a row that
fails to match can neither mask nor detect its identifier, so a narrowed or broken row scores a
vacuous pass while the raw value survives in the output. `score_pii_safety` scans each surface
two independent ways:

1. `pack_leak` uses the same rows the redactor uses. Catches PII the pipeline re-introduced
   after redaction; blind by construction to the pack being wrong.
2. `planted_leak` looks for a test's own planted identifier as a literal, with no pack involved.
   Against the real redactor this is a sound oracle: narrow, mis-escape or delete a row and the
   redactor stops masking that market *and* `pack_leak` stops detecting it, so only this check
   still fails.

## RE2-safe forms

Some matchers (for example Google Cloud DLP custom info types) use RE2, which has no lookaround.
`re2_pattern_for(info_type, pattern)` returns a lookaround-free source for such rows, so a
managed profile cannot be handed an inspect config it will reject. A test pins that no RE2 form
contains lookaround.

## Separator normalisation

Validators strip `[\s-]` (any whitespace or hyphen) before checking. Regex rows admit `[\s-]`
as a separator, and PDF text extraction routinely emits non-breaking spaces and tabs; a value a
row matches but a validator cannot normalise would fail `isdigit()` and leak undetected. The
normalisation closes that seam, and a test pins it.

## Development

```sh
pip install -e ".[dev]"     # ruff (pinned), mypy, pytest
ruff check src tests
ruff format --check src tests
mypy src
pytest
```

The model channel is tested **hermetically** — a scripted stdlib HTTP server stands in for the
endpoint, so `pytest` needs no GPU, no model, and no network. CI never talks to a real model.

## Notes

- All example identifiers in the code and tests are obviously fictional.
- Checksum validators reduce but do not eliminate false positives (a small fraction of random
  digit runs pass any mod-N check); the residual is pinned by a test rather than hidden.

## License

Apache-2.0.
