"""Checksum validators: known-good, known-bad, the separator seam, and the pinned residual."""

from __future__ import annotations

import pytest

from pii_kit import validators as v


class TestKnownValid:
    def test_sg_nric(self) -> None:
        assert v.sg_nric_valid("S1234567D")  # valid check letter
        assert v.sg_nric_valid("s1234567d"), "case-insensitive"
        assert not v.sg_nric_valid("S1234567A"), "wrong check letter"
        assert not v.sg_nric_valid("X1234567D"), "bad prefix"

    def test_hk_hkid(self) -> None:
        assert v.hk_hkid_valid("A123456(3)")
        assert v.hk_hkid_valid("A1234563"), "bare keyed form still validates"
        assert not v.hk_hkid_valid("A123456(4)"), "wrong check char"

    def test_jp_my_number(self) -> None:
        assert v.jp_my_number_valid("123456789018")
        assert v.jp_my_number_valid("1234 5678 9018"), "grouped printed form"
        assert not v.jp_my_number_valid("123456789019"), "wrong check digit"
        assert not v.jp_my_number_valid("111111111111"), "all-identical rejected"

    def test_au_tfn(self) -> None:
        assert v.au_tfn_valid("123456782")
        assert v.au_tfn_valid("123 456 782"), "spaced form"
        assert not v.au_tfn_valid("123456783"), "wrong checksum"

    def test_au_abn(self) -> None:
        assert v.au_abn_valid("51824753556")
        assert v.au_abn_valid("51 824 753 556")
        assert not v.au_abn_valid("51824753557")

    def test_au_medicare(self) -> None:
        assert v.au_medicare_valid("2123456701")
        # The check digit is the 9th (index 8); break it, not the trailing issue digit.
        assert not v.au_medicare_valid("2123456711"), "wrong check digit"

    def test_luhn(self) -> None:
        assert v.luhn_valid("4111111111111111")
        assert not v.luhn_valid("4111111111111112")
        assert not v.luhn_valid("0000000000000000"), "degenerate run rejected"

    def test_iban(self) -> None:
        assert v.iban_valid("GB82 WEST 1234 5698 7654 32")
        assert not v.iban_valid("GB82WEST12345698765433")


class TestSeparatorSeam:
    """Separators are stripped with ``[\\s-]`` (any whitespace or hyphen), not just ``[ -]``.

    PDF text extraction routinely emits a non-breaking space, and these redactors run over
    parser output. A value the regex row admits but a validator cannot normalise fails
    ``isdigit()`` and leaks. Pin that a non-breaking / tab separator still validates.
    """

    @pytest.mark.parametrize("sep", [" ", "\t", "\n"])
    def test_tfn_with_whitespace_separator(self, sep: str) -> None:
        assert v.au_tfn_valid(f"123{sep}456{sep}782")

    @pytest.mark.parametrize("sep", [" ", "\t"])
    def test_my_number_with_whitespace_separator(self, sep: str) -> None:
        assert v.jp_my_number_valid(f"1234{sep}5678{sep}9018")


class TestResidualPinned:
    """The checksum false-positive residual, pinned rather than hidden.

    A validator is a filter, not a proof of identity: ~1 in 11 random nine-digit runs passes
    the TFN mod-11 by chance, round numbers especially. Pin a KNOWN chance-passing run so a
    future reader meets the residual as a decision, and pin that a bare shape does NOT imply a
    valid identifier.
    """

    def test_round_number_passes_tfn_by_chance(self) -> None:
        # 000000000 trivially satisfies sum % 11 == 0; documents that the checksum bounds but
        # does not eliminate false positives.
        assert v.au_tfn_valid("000000000")

    def test_residual_rate_is_roughly_one_in_eleven(self) -> None:
        hits = sum(1 for n in range(100000) if v.au_tfn_valid(f"{n:09d}"))
        rate = hits / 100000
        assert 0.06 < rate < 0.12, f"TFN checksum pass-rate {rate} outside pinned band"
