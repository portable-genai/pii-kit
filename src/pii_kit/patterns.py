"""Jurisdiction PII pattern rows: a shared, versioned source a redactor and a leak check both
read from.

Each row is a 3-tuple ``(info_type, re.Pattern, validator | None)``. ``validator`` (when
present) is a ``str -> bool`` predicate applied to each raw match, so a redactor masks and a
leak-check detects a match only when it is a genuine identifier. Reading the rows and the
validators from one place is what stops a bespoke detector and a bespoke redactor drifting
apart.

What is in this package
-----------------------
The jurisdiction national-identifier rows (SG / HK / JP / AU, plus reference IN / GB), the
universal ``EMAIL`` and ``PHONE_NUMBER`` rows, the checksum validators (in
:mod:`pii_kit.validators`), and an RE2-safe source per row for consumers whose matcher has no
lookaround (for example Google Cloud DLP custom info types).

What is deliberately NOT in this package
----------------------------------------
- **Row ORDER.** The application order is application-specific, not a constant: a bare-digit
  account catch-all (``\\b\\d{9,17}\\b``) SUBSUMES the contiguous national-id shapes and must run
  LAST, whereas a specific 3-6-1 statement account shape (``\\b\\d{3}-\\d{6}-\\d\\b``) is bitten
  INTO by the AU TFN row and must run FIRST. The package exposes the rows and the universal
  rows; each consumer assembles and orders the full list around its own account row. Use
  :func:`national_patterns_for` for the national rows and compose with the universal rows and
  your account row in the order your account shape dictates.
- **The account-number row.** It is application-specific PII with an application-specific shape
  (see above), so it lives in the consuming application, not here.
- **The redaction call sites and any test fixtures.** Both are application-specific.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from . import validators

#: A single redaction rule: an info type, the regex that finds it, and an optional checksum
#: validator applied to each raw match before it is treated as a real identifier.
Pattern = tuple[str, re.Pattern[str], Callable[[str], bool] | None]


# --------------------------------------------------------------------------- universal rows
# Applied for every jurisdiction. Email and phone are universal PII; the account-number row
# is NOT here because its shape is application-specific (see the module docstring).

EMAIL: Pattern = (
    "EMAIL_ADDRESS",
    re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    None,
)
PHONE_INTL: Pattern = (
    "PHONE_NUMBER",
    re.compile(r"\+\d{1,3}[\s-]?\d(?:[\s-]?\d){6,13}\b"),
    None,
)

#: The universal rows, in their canonical leading order.
UNIVERSAL_PATTERNS: tuple[Pattern, ...] = (EMAIL, PHONE_INTL)


# ---------------------------------------------------------------- per-jurisdiction national
#: ISO-3166 alpha-2 -> the national-identifier rows for that market. Three rows are written to
#: avoid common narrowing bugs: ``JP_MY_NUMBER`` as a bare ``\\b\\d{12}\\b`` would miss the
#: grouped ``1234 5678 9018`` form a My Number card is printed in; an upper-case-only
#: ``SG_NRIC_FIN`` would miss a lower-cased NRIC typed into a free-text field; an ``HK_HKID`` row
#: requiring the parens would leak the bare keyed ``A1234563``. All three forms are covered here.
NATIONAL_ID_PATTERNS: dict[str, list[Pattern]] = {
    "SG": [
        # Case-insensitive. No checksum: the shape does not collide with ordinary text, and
        # recall at the boundary beats checksum precision.
        ("SG_NRIC_FIN", re.compile(r"\b[STFGMstfgm]\d{7}[A-Za-z]\b"), None),
        ("SG_PHONE", re.compile(r"\b(?:\+?65[\s-]?)?[689]\d{3}[\s-]?\d{4}\b"), None),
    ],
    "HK": [
        # Two rows: the parenthesised form is unambiguous and masked on shape; the bare keyed
        # form collides with ordinary document references (e.g. AB1234567), so it is
        # checksum-gated instead of leaking or eating every reference number.
        ("HK_HKID", re.compile(r"\b[A-Z]{1,2}\d{6}\([0-9A]\)"), None),
        ("HK_HKID", re.compile(r"\b[A-Z]{1,2}\d{6}[0-9A]\b"), validators.hk_hkid_valid),
    ],
    "JP": [
        # Checksum-gated so a genuine My Number is reported as one; a 12-digit run that fails
        # the check is an account/reference number. The 4-4-4 grouping is the printed form,
        # which a bare-digit account row cannot see. The leading lookarounds also reject a
        # 12-digit prefix of a longer grouped run (e.g. the first 12 digits of a 16-digit card
        # PAN, which can pass the My Number checksum by chance).
        (
            "JP_MY_NUMBER",
            re.compile(r"(?<!\d)(?<!\d[- ])\d{4}[- ]?\d{4}[- ]?\d{4}(?![- ]?\d)"),
            validators.jp_my_number_valid,
        ),
    ],
    "AU": [
        # A 9-digit run. The gate decides whether the SPACED form is caught at all; a
        # contiguous run a bare-digit account row would also see.
        ("AU_TFN", re.compile(r"\b\d{3}[\s-]?\d{3}[\s-]?\d{3}\b"), validators.au_tfn_valid),
    ],
    "IN": [
        ("IN_PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
        ("IN_AADHAAR", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), None),
    ],
    "GB": [
        # National Insurance number (prefix rules simplified to a safe superset).
        ("GB_NINO", re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b"), None),
    ],
}

#: Reference APAC default. Consumers select their own set; this is only a convenience.
DEFAULT_JURISDICTIONS: tuple[str, ...] = ("SG", "HK", "JP", "AU")


# --------------------------------------------------------------------------------- RE2 forms
#: RE2-safe equivalents, by info type, for rows whose Python regex uses syntax RE2 rejects.
#: Google Cloud DLP custom info types are matched with RE2, which has no lookaround, so the JP
#: row's lookarounds make DLP reject the whole inspect config with INVALID_ARGUMENT: the
#: managed profile would fail on every call rather than degrade. RE2 supports ``\b``, which
#: still rejects a 12-digit prefix of a longer CONTIGUOUS run; what is lost is only the
#: rejection of a 12-digit prefix of a longer GROUPED run, which merely masks more (the same
#: fail-safe direction as the missing checksum). Both forms live here so they cannot drift.
_RE2_OVERRIDES: dict[str, str] = {
    "JP_MY_NUMBER": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b",
}


def re2_pattern_for(info_type: str, pattern: re.Pattern[str]) -> str:
    """The RE2-compatible source for a row, for consumers that cannot run Python regex."""
    return _RE2_OVERRIDES.get(info_type, pattern.pattern)


def national_patterns_for(jurisdictions: Iterable[str]) -> list[Pattern]:
    """The national-identifier rows for ``jurisdictions``, de-duplicated, WITHOUT ordering
    them against any account row.

    The order WITHIN the returned list follows :data:`NATIONAL_ID_PATTERNS`; where the whole
    list sits relative to your universal rows and your account row is your decision, because
    that order is application-specific (see the module docstring). Compose, for a
    bare-digit-catch-all account application::

        rows = [*UNIVERSAL_PATTERNS, *national_patterns_for(js), account_row]  # account LAST

    and for a specific-shape account application the AU TFN row bites into::

        rows = [*UNIVERSAL_PATTERNS, account_row, *national_patterns_for(js)]  # account FIRST

    Unknown jurisdiction codes contribute no rows (they are skipped, not an error), so a
    partially-configured fork degrades to the universal rows rather than raising.

    De-duplication is keyed on the (info type, regex source) pair, not the info type alone: two
    jurisdictions may share a row, but one jurisdiction may also carry the SAME info type under
    two shapes (HK's parenthesised and bare HKID forms), and keying on the name alone would
    silently drop the second.
    """
    national: list[Pattern] = []
    seen: set[tuple[str, str]] = set()
    for code in jurisdictions:
        for row in NATIONAL_ID_PATTERNS.get((code or "").upper(), ()):
            key = (row[0], row[1].pattern)
            if key not in seen:
                national.append(row)
                seen.add(key)
    return national
