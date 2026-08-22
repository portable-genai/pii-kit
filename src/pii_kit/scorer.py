"""A two-part PII-safety scorer for tests and gates.

A leak-check scored ONLY off the same pack a redactor masks with is a closed loop: a row that
fails to match cannot detect what it failed to mask, so a narrowed or broken row scores a
vacuous pass while the raw identifier survives in the output. The fix is two independent
halves, both provided here:

1. :func:`pack_leak` scans with the SAME rows the redactor uses. This catches PII the pipeline
   RE-INTRODUCED after redaction, and is blind by construction to anything the pack itself
   gets wrong.
2. :func:`planted_leak` scans for a specific identifier a test case planted, as a literal, with
   no pack involved. Against the real redactor this is a sound oracle the pack has no say in:
   narrow, mis-escape or delete a row and the redactor silently stops masking that market AND
   :func:`pack_leak` silently stops detecting it, so only this check still fails.

:func:`score_pii_safety` combines them. Feed it DERIVED surfaces (records that outlive the
request, the text a model reads), never an echo of the caller's raw input, or the metric
measures the fixture instead of the redaction boundary.

An OPTIONAL third channel, :func:`pii_kit.model.model_leak`, joins when the caller passes a
``model_config``: a locally hosted language model scans each surface for the PII no regex row
can express (names, addresses, free text). It is advisory and red-only — a model finding can
drop the score to 0.0, never lift it — so the two deterministic halves keep their guarantees
with or without it.

:func:`redact` is a small, optional convenience so a consumer's redactor and this scorer share
one masking definition; WHERE in the pipeline a consumer redacts is still its own concern.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from .patterns import Pattern

if TYPE_CHECKING:
    from .model import ModelConfig

#: Default replacement token, using a conventional ``[REDACTED:<type>]`` marker.
DEFAULT_MASK = "[REDACTED:{info_type}]"


def pack_leak(text: str, patterns: Iterable[Pattern]) -> bool:
    """True if any pattern (validator-gated) still matches ``text``.

    The pack-DEPENDENT half of the check. Detects exactly what the redactor masks, so it can
    only catch PII the pipeline re-introduced after redaction; pair it with
    :func:`planted_leak`, which is what catches the pack being wrong.
    """
    haystack = text or ""
    for _info_type, pattern, validator in patterns:
        for match in pattern.finditer(haystack):
            if validator is None or validator(match.group(0)):
                return True
    return False


def planted_leak(text: str, planted_tokens: Iterable[str]) -> bool:
    """True if any of ``planted_tokens`` survives ``text`` verbatim.

    The pack-INDEPENDENT half. ``planted_tokens`` are the identifiers a test case injected
    (a national id in its printed form, plus any universal fixtures such as the account number
    and email). A literal substring test the pack cannot influence: it fails precisely when a
    row is narrowed, mis-escaped or deleted and the redactor stops masking that market.
    """
    haystack = text or ""
    return any(token and token in haystack for token in planted_tokens)


def score_pii_safety(
    haystacks: Iterable[str],
    patterns: Iterable[Pattern],
    planted_tokens: Iterable[str] = (),
    model_config: ModelConfig | None = None,
) -> float:
    """1.0 unless unredacted PII survived any of ``haystacks``; 0.0 if anything leaked.

    Scans each surface both ways (pack scan + planted-literal scan). A single survivor drops
    the metric to 0.0, so the gate fails if anything bypassed the redact-before-everything
    boundary. ``patterns`` must be materialised once by the caller (do not recompute per call);
    ``planted_tokens`` may be empty for a case that plants no PII, in which case only the pack
    scan runs.

    ``model_config`` (optional) adds the model-assisted third scan over the same surfaces —
    red-only, so it can only lower the score, and run after the deterministic halves so it
    costs nothing when they already failed. A dead endpoint raises
    :class:`pii_kit.model.ModelAPIError` rather than skipping the advertised scan silently.
    """
    patterns = list(patterns)
    planted = list(planted_tokens)
    surfaces = list(haystacks)
    leaked = any(pack_leak(hay, patterns) or planted_leak(hay, planted) for hay in surfaces)
    if not leaked and model_config is not None:
        # Imported at call time: pii_kit.model imports this module (DEFAULT_MASK), so a
        # module-level import here would be a cycle.
        from .model import model_leak

        leaked = any(model_leak(hay, model_config) for hay in surfaces)
    return 0.0 if leaked else 1.0


def redact(text: str, patterns: Iterable[Pattern], mask: str = DEFAULT_MASK) -> str:
    """Mask every validated match of ``patterns`` in ``text`` (optional convenience).

    Applies the rows IN THE GIVEN ORDER, so the caller controls precedence (which matters when
    an account row and a national-id row can claim overlapping digit runs; see
    :func:`pii_kit.patterns.national_patterns_for`). ``mask`` may reference ``{info_type}``.
    Redacting the same rows this module scores means the scorer's pack half and the redactor
    cannot drift; keeping it optional means a repo with its own redaction adapter still shares
    the one pattern source without adopting a masking style it did not choose.
    """
    out = text or ""
    for info_type, pattern, validator in patterns:
        replacement = mask.format(info_type=info_type)

        def _sub(
            m: re.Match[str],
            _rep: str = replacement,
            _val: Callable[[str], bool] | None = validator,
        ) -> str:
            value = m.group(0)
            if _val is not None and not _val(value):
                return value
            return _rep

        out = pattern.sub(_sub, out)
    return out
