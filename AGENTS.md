# pii-kit

The shared working agreement is [`.github/AGENTS.md`](https://github.com/portable-genai/.github/blob/main/AGENTS.md).
It carries the architecture rules, the gate contract, the fleet invariants, the
falsification discipline, versions and house style, and it holds in every repository
here. Read it first. This file carries only what is specific to this one.

## Commands

A venv already exists at `.venv` with the package installed editable. Setup from scratch:

```sh
pip install -e ".[dev]"        # ruff (pinned exactly), mypy, pytest
```

The full CI gate, in order (all four must pass):

```sh
ruff check src tests
ruff format --check src tests   # ruff is pinned EXACTLY in pyproject.toml so formatting never drifts
mypy src                        # strict mode; src only, tests are not type-checked
pytest                          # -q, testpaths=tests (from pyproject)
```

Run a single test:

```sh
pytest tests/test_scorer.py::TestPerMarketNotFalselyGreen -q
pytest tests/test_patterns.py -k grouped_form -q
```

## Hard constraints

- **Zero runtime dependencies, pure stdlib.** This is the package's core promise (installs on an air-gapped host). Never add a runtime dependency; `dependencies = []` in pyproject.toml is deliberate. Dev tooling goes in the `dev` extra only.
- **Python ≥3.12**, mypy `strict = true`, ruff line-length 100 with `E,F,I,UP,B,SIM`.
- For checksum-gated rows (JP My Number, AU TFN), fictional test values still need **valid check digits** or the row won't fire (see `PLANTED` in tests/test_scorer.py).

## Architecture

Four modules in `src/pii_kit/`, re-exported flat from `__init__.py`:

- **patterns.py**: the pattern rows. A row is `Pattern = (info_type, re.Pattern, validator | None)`. `NATIONAL_ID_PATTERNS` maps ISO-3166 alpha-2 → rows; `UNIVERSAL_PATTERNS` is `(EMAIL, PHONE_INTL)`. `national_patterns_for()` de-duplicates keyed on **(info_type, regex source)**, not info_type alone, because HK carries two `HK_HKID` rows (parenthesised shape-only + bare checksum-gated). `re2_pattern_for()` returns a lookaround-free source per row (`_RE2_OVERRIDES`) for RE2 matchers like Google Cloud DLP; a test pins that no RE2 form contains lookaround.
- **validators.py**: pure checksum predicates (`str -> bool`) that gate regex matches into real detections. Validators strip `[\s-]` (any whitespace incl. non-breaking, or hyphen) before checking. That is the "separator seam": a value the regex matches but the validator can't normalise would silently leak. Pinned by a test.
- **scorer.py**: the two-part safety scorer. `pack_leak` scans with the same rows the redactor uses (catches re-introduced PII, blind to a broken pack); `planted_leak` is a pack-independent literal-substring oracle (catches a narrowed/mis-escaped/deleted row). `score_pii_safety` combines them: 1.0 clean, 0.0 on any leak. It also takes an optional `model_config` that adds the model channel as a **third** scan (run after the two deterministic halves, so it costs nothing once they've already gone red). `redact()` is an optional convenience that applies rows **in the given order**.
- **model.py**: the OPTIONAL model-assisted channel (setup guide: `docs/vllm-gemma-setup.md`) (off unless a `ModelConfig` is supplied; `ModelConfig.from_env()` reads `PII_PACK_MODEL_BASE_URL` / `PII_PACK_MODEL` / `PII_PACK_MODEL_API_KEY`). Calls any OpenAI-compatible chat-completions endpoint (vLLM, Ollama `/v1`, llama-server) via **stdlib `urllib` only**; never add the `openai` package. Design rules that must survive edits: the model is **advisory and red-only** (a finding can lower `score_pii_safety` to 0.0, never lift it; the pack always runs and cannot be overridden); text findings are **grounded** (verbatim-substring-checked) and only grounded findings may be masked by `model_redact`; image findings are inherently ungrounded, detect-only; endpoint failure or an unparseable reply **raises `ModelAPIError`** and never degrades to "no findings", which would score a vacuous pass; error messages carry the **HTTP status only**, never the scanned input and never the server's response body (an echoing server/proxy could fill it with the scanned request); `model_redact` masks **span-based against the original text** (overlaps merged, deterministic labels), never sequential `str.replace`; requests use a **single user turn** (Gemma chat templates have no system role). `model.py` imports `scorer.DEFAULT_MASK`, so scorer imports `model_leak` lazily inside `score_pii_safety`; do not "fix" that into a module-level cycle.

### Deliberate non-goals (do not "fix" these)

- **Row order is not baked in.** The right order is application-specific: a bare-digit account catch-all must run last (it subsumes national-id shapes), a specific-shape account row must run first (the AU TFN row bites into it). Consumers compose.
- **No account-number row.** Account shapes are application-specific PII; they live in the consuming application.

### Why the scorer has two halves

A leak check scored only off the redactor's own rows is a closed loop: a broken row can neither mask nor detect, so it scores a vacuous pass. The planted-literal half is the oracle that fails in exactly that case. Keep both halves independent; don't route `planted_leak` through the pack.

### Load-bearing test invariants

Tests pin design decisions, not just behaviour, so read the test docstrings before changing a row. Key pins: JP My Number matches the printed grouped `1234 5678 9018` form (and its lookarounds reject 12-digit prefixes of longer runs); SG NRIC is case-insensitive; HK's bare keyed form matches but is checksum-gated; per-market "goes red with redaction disabled" proves the metric can fail; the residual checksum false-positive rate is pinned rather than hidden.

Model-channel tests (tests/test_model.py) are hermetic: a scripted stdlib `http.server` (`MockModelServer`) plays the endpoint, and CI never talks to a real model. Pins: no `ModelConfig` ⇒ no HTTP call at all; a red deterministic score short-circuits before the model runs (so the model cannot influence an already-red score); every surface is scanned (not just the first); a dead endpoint raises through the gate; error text never echoes the scanned input **or** the server's error body; `model_redact` masks span-based (a value inside an inserted mask token can't corrupt it, partial overlaps merge, identical text under two types masks deterministically).
