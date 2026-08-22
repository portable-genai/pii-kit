"""pii-kit: a shared, versioned jurisdiction PII pattern pack.

One source of truth for what a national identifier LOOKS like (SG / HK / JP / AU national ids,
universal email / phone), the checksum validators that harden those shapes, RE2-safe forms for
matchers with no lookaround, and a two-part PII-safety scorer that cannot go falsely green.
Reading the rows and validators from one place lets a redactor and its test-time leak check
share the same definition, so a fix is a version bump rather than an N-place copy that drifts.

Pure stdlib, zero runtime dependencies: it installs and runs on an air-gapped host.

An OPTIONAL model-assisted channel (:mod:`pii_kit.model`) can raise recall on free-text PII
and scan images, by calling a locally hosted OpenAI-compatible endpoint (e.g. Gemma on vLLM;
see ``docs/vllm-gemma-setup.md``). It is off unless a :class:`~pii_kit.model.ModelConfig` is
supplied, advisory (it only ever ADDS detections), and itself stdlib-only.

Row ORDER and the application-specific account-number row are deliberately NOT here (see
:mod:`pii_kit.patterns`); order is application-specific, so each consumer composes and orders
the rows around its own account shape.
"""

from __future__ import annotations

from . import model, patterns, scorer, validators
from .model import (
    DEFAULT_MODEL_INFO_TYPES,
    ModelAPIError,
    ModelConfig,
    ModelFinding,
    model_image_findings,
    model_image_leak,
    model_leak,
    model_redact,
    model_text_findings,
)
from .patterns import (
    DEFAULT_JURISDICTIONS,
    EMAIL,
    NATIONAL_ID_PATTERNS,
    PHONE_INTL,
    UNIVERSAL_PATTERNS,
    Pattern,
    national_patterns_for,
    re2_pattern_for,
)
from .scorer import pack_leak, planted_leak, redact, score_pii_safety

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_JURISDICTIONS",
    "DEFAULT_MODEL_INFO_TYPES",
    "EMAIL",
    "ModelAPIError",
    "ModelConfig",
    "ModelFinding",
    "NATIONAL_ID_PATTERNS",
    "PHONE_INTL",
    "Pattern",
    "UNIVERSAL_PATTERNS",
    "__version__",
    "model",
    "model_image_findings",
    "model_image_leak",
    "model_leak",
    "model_redact",
    "model_text_findings",
    "national_patterns_for",
    "pack_leak",
    "patterns",
    "planted_leak",
    "re2_pattern_for",
    "redact",
    "score_pii_safety",
    "scorer",
    "validators",
]
