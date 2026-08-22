"""Checksum validators for the jurisdiction PII pack: pure-stdlib, zero dependencies.

Each validator takes the raw matched string and returns ``True`` only when it is a
genuine identifier of that type. A validator PASS is what lifts a format-only regex match
to a real detection; a validator FAIL kills the candidate, which is what keeps false
positives low without any ML.

Separator handling
------------------
Validators strip ``[\\s-]`` (any whitespace or hyphen) before checking, because the regex
rows in :mod:`pii_kit.patterns` admit ``[\\s-]`` as a separator. If a row can match a value
that a validator then cannot normalise, that value fails ``isdigit()`` and is neither masked
nor detected: a silent leak. The canonical case is a TFN or My Number separated by a
non-breaking space, which PDF text extraction routinely emits and redactors run over parser
output. Normalising to ``[\\s-]`` closes that seam, and it is pinned by a test.
"""

from __future__ import annotations

import re

_SEP = re.compile(r"[\s-]")


def luhn_valid(digits: str) -> bool:
    """Luhn mod-10 check for a bare digit string (PAN, 13-19 digits)."""
    if not digits.isdigit() or len(digits) < 13:
        return False
    if len(set(digits)) == 1:
        return False  # degenerate all-identical run is never a real PAN
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_NRIC_ST = "JZIHGFEDCBA"
_NRIC_FG = "XWUTRQPNMLK"
_NRIC_M = "KLJNPQRTUWX"
_NRIC_WEIGHTS = (2, 7, 6, 5, 4, 3, 2)


def sg_nric_valid(value: str) -> bool:
    """Singapore NRIC / FIN: prefix + 7 digits + check letter (case-insensitive)."""
    value = value.upper()
    if len(value) != 9 or value[0] not in "STFGM" or not value[1:8].isdigit():
        return False
    s = sum(int(d) * w for d, w in zip(value[1:8], _NRIC_WEIGHTS, strict=True))
    prefix = value[0]
    if prefix in "TG":
        s += 4
    elif prefix == "M":
        s += 3
    table = _NRIC_ST if prefix in "ST" else _NRIC_FG if prefix in "FG" else _NRIC_M
    return table[s % 11] == value[8]


def hk_hkid_valid(value: str) -> bool:
    """Hong Kong Identity Card: 1-2 prefix letters + 6 digits + check char."""
    m = re.fullmatch(r"([A-Z]{1,2})(\d{6})\(?([0-9A])\)?", value.upper())
    if not m:
        return False
    prefix, digits, check = m.groups()
    vals = [36, ord(prefix) - 55] if len(prefix) == 1 else [ord(c) - 55 for c in prefix]
    vals.extend(int(c) for c in digits)
    s = sum(v * w for v, w in zip(vals, range(9, 1, -1), strict=True))
    r = (11 - s % 11) % 11
    return check == ("A" if r == 10 else str(r))


def jp_my_number_valid(value: str) -> bool:
    """Japanese Individual Number (My Number): 12 digits with a trailing check digit."""
    digits = _SEP.sub("", value)
    if not digits.isdigit() or len(digits) != 12:
        return False
    if len(set(digits)) == 1:
        return False  # an all-identical run is not a real My Number
    s = 0
    for n in range(1, 12):
        p = int(digits[11 - n])  # P_n: nth digit from the right of the first 11
        q = n + 1 if n <= 6 else n - 5
        s += p * q
    r = s % 11
    return int(digits[11]) == (0 if r <= 1 else 11 - r)


def au_tfn_valid(value: str) -> bool:
    """Australian Tax File Number: 9 digits, weighted checksum mod 11."""
    digits = _SEP.sub("", value)
    if not digits.isdigit() or len(digits) != 9:
        return False
    weights = (1, 4, 3, 7, 5, 8, 6, 9, 10)
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11 == 0


def au_abn_valid(value: str) -> bool:
    """Australian Business Number: 11 digits, subtract-1-from-first, weighted mod 89."""
    digits = re.sub(r"\s", "", value)
    if not digits.isdigit() or len(digits) != 11 or digits[0] == "0":
        return False
    weights = (10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
    ds = [int(c) for c in digits]
    ds[0] -= 1
    return sum(d * w for d, w in zip(ds, weights, strict=True)) % 89 == 0


def au_medicare_valid(value: str) -> bool:
    """Australian Medicare card: 10-11 digits, first in 2-6, weighted mod 10 check."""
    digits = re.sub(r"[\s/]", "", value)
    if not digits.isdigit() or len(digits) not in (10, 11) or digits[0] not in "23456":
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9)
    return sum(int(d) * w for d, w in zip(digits[:8], weights, strict=True)) % 10 == int(digits[8])


def iban_valid(value: str) -> bool:
    """IBAN: mod-97 check over the rearranged, letter-substituted string."""
    v = re.sub(r"\s", "", value.upper())
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", v):
        return False
    rearranged = v[4:] + v[:4]
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    return int(digits) % 97 == 1
