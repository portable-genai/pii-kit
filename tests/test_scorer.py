"""The two-part scorer: per-market not-falsely-green, and the broken-row oracle.

These are the load-bearing tests. A PII-safety metric scored ONLY off the pack cannot detect
the pack being wrong, so the pack-independent literal check is what makes the per-market claim
real.
"""

from __future__ import annotations

import re

import pytest

from pii_kit import (
    UNIVERSAL_PATTERNS,
    Pattern,
    national_patterns_for,
    pack_leak,
    planted_leak,
    redact,
    score_pii_safety,
)

# One obviously-fictional identifier per market, in the form the card / notice prints. JP and
# AU carry VALID check digits because their rows are checksum-gated. They are what proves each
# market's row actually masks.
PLANTED: dict[str, str] = {
    "SG": "S1234567D",
    "HK": "A123456(3)",
    "JP": "1234 5678 9018",
    "AU": "123 456 782",
}
MARKETS = ("SG", "HK", "JP", "AU")


def _rows(jurisdictions: tuple[str, ...]) -> list[Pattern]:
    # A specific-shape account application: account row is NOT a bare-digit catch-all, so
    # national rows can follow the universal rows. Order chosen by the consumer.
    return [*UNIVERSAL_PATTERNS, *national_patterns_for(jurisdictions)]


class TestPerMarketNotFalselyGreen:
    """Prove per market that redaction is exercised and the metric CAN go red."""

    @pytest.mark.parametrize("market", MARKETS)
    def test_market_goes_red_with_redaction_disabled(self, market: str) -> None:
        rows = _rows((market,))
        planted = PLANTED[market]
        text = f"Applicant {market} id {planted} on file."
        # Redaction DISABLED: the raw identifier is still present, so the scorer must fail.
        score = score_pii_safety([text], rows, planted_tokens=[planted])
        assert score == 0.0, f"{market}: scorer falsely green with redaction disabled"

    @pytest.mark.parametrize("market", MARKETS)
    def test_market_passes_after_real_redaction(self, market: str) -> None:
        rows = _rows((market,))
        planted = PLANTED[market]
        text = f"Applicant {market} id {planted} on file."
        redacted = redact(text, rows)
        assert planted not in redacted, f"{market}: identifier survived redaction"
        score = score_pii_safety([redacted], rows, planted_tokens=[planted])
        assert score == 1.0, f"{market}: scorer red after correct redaction"


class TestBrokenRowIsCaught:
    """With a market's row broken, the pack scan goes blind but the literal check fires.

    This is the whole point of the two-part scorer: a row that fails to match
    can neither mask nor detect its market, so the pack half scores a vacuous pass while the
    raw identifier sits in the output. Only the pack-INDEPENDENT literal check still fails.
    """

    def test_broken_jp_row_defeats_pack_scan_but_not_literal_check(self) -> None:
        planted = PLANTED["JP"]
        # Simulate a narrowed / deleted JP row: a pattern that never matches the grouped form.
        broken: list[Pattern] = [
            *UNIVERSAL_PATTERNS,
            ("JP_MY_NUMBER", re.compile(r"\bZZZ_NEVER_MATCHES\b"), None),
        ]
        text = f"My Number {planted} recorded."
        # A redactor built from the broken pack cannot mask it...
        redacted = redact(text, broken)
        assert planted in redacted, "broken row unexpectedly masked the identifier"
        # ...and the pack scan is correspondingly blind...
        assert pack_leak(redacted, broken) is False
        # ...but the literal oracle catches it, so the combined score is red.
        assert planted_leak(redacted, [planted]) is True
        assert score_pii_safety([redacted], broken, planted_tokens=[planted]) == 0.0


class TestScorerContract:
    def test_empty_haystacks_pass(self) -> None:
        assert score_pii_safety([], _rows(MARKETS)) == 1.0

    def test_no_planted_tokens_runs_pack_scan_only(self) -> None:
        rows = _rows(("AU",))
        # A re-introduced (un-redacted) valid TFN is caught by the pack scan with no fixture.
        assert score_pii_safety(["TFN 123 456 782"], rows) == 0.0
        assert score_pii_safety(["nothing sensitive here"], rows) == 1.0

    def test_planted_leak_ignores_empty_tokens(self) -> None:
        assert planted_leak("anything", ["", None]) is False  # type: ignore[list-item]

    def test_redact_respects_given_order(self) -> None:
        # Account-first vs national-first changes which info type claims an overlapping run.
        account: Pattern = ("BANK_ACCOUNT_NUMBER", re.compile(r"\b\d{3}-\d{6}-\d\b"), None)
        au_rows = national_patterns_for(("AU",))
        text = "account 123-456782-0 here"
        account_first = redact(text, [account, *au_rows])
        assert "[REDACTED:BANK_ACCOUNT_NUMBER]" in account_first
        assert "AU_TFN" not in account_first
