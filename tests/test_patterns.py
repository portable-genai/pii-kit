"""The rows themselves: the three fixes over the narrowed sibling packs, and dedup behaviour."""

from __future__ import annotations

import re

from pii_kit import patterns as p
from pii_kit.patterns import national_patterns_for


def _rows_matching(info_type: str, jurisdiction: str) -> list[p.Pattern]:
    return [row for row in p.NATIONAL_ID_PATTERNS[jurisdiction] if row[0] == info_type]


class TestThreeFixes:
    """The three rows most prone to being narrowed into silent leaks."""

    def test_jp_my_number_catches_grouped_form(self) -> None:
        # The narrowed \b\d{12}\b row could not see the printed 4-4-4 form at all.
        _info, pattern, validator = _rows_matching("JP_MY_NUMBER", "JP")[0]
        m = pattern.search("My Number 1234 5678 9018 on file")
        assert m is not None
        assert validator is not None and validator(m.group(0))

    def test_sg_nric_is_case_insensitive(self) -> None:
        _info, pattern, _validator = _rows_matching("SG_NRIC_FIN", "SG")[0]
        assert pattern.search("nric s1234567d") is not None, "lower-case NRIC must match"

    def test_hk_bare_keyed_form_matches_and_is_gated(self) -> None:
        # The row requiring parens leaked the bare keyed A1234563. There must be a row that
        # matches the bare form, gated by the checksum.
        bare = [
            (i, pat, val)
            for (i, pat, val) in p.NATIONAL_ID_PATTERNS["HK"]
            if val is not None and pat.search("A1234563")
        ]
        assert bare, "no checksum-gated row matches the bare keyed HKID form"


class TestNationalPatternsFor:
    def test_unknown_code_is_skipped_not_raised(self) -> None:
        assert national_patterns_for(["ZZ"]) == []

    def test_lowercase_and_blank_codes(self) -> None:
        assert national_patterns_for(["sg"]) == national_patterns_for(["SG"])
        assert national_patterns_for(["", None]) == []  # type: ignore[list-item]

    def test_hk_two_shapes_of_same_info_type_both_kept(self) -> None:
        # Dedup keys on (info_type, regex source), so HK's two HK_HKID rows both survive.
        rows = national_patterns_for(["HK"])
        assert sum(1 for r in rows if r[0] == "HK_HKID") == 2

    def test_dedup_across_jurisdictions_is_by_regex_not_name(self) -> None:
        rows = national_patterns_for(["SG", "SG"])
        sources = [(r[0], r[1].pattern) for r in rows]
        assert len(sources) == len(set(sources)), "duplicate rows not de-duplicated"

    def test_order_is_not_baked_in(self) -> None:
        # The package must NOT return universal or account rows; ordering is the consumer's.
        rows = national_patterns_for(p.DEFAULT_JURISDICTIONS)
        infos = {r[0] for r in rows}
        assert "EMAIL_ADDRESS" not in infos
        assert "PHONE_NUMBER" not in infos
        assert "BANK_ACCOUNT_NUMBER" not in infos


class TestUniversalRows:
    def test_email_matches(self) -> None:
        _info, pattern, _v = p.EMAIL
        assert pattern.search("reach me at a.b+x@example.co.uk please")

    def test_phone_intl_matches(self) -> None:
        _info, pattern, _v = p.PHONE_INTL
        assert pattern.search("call +65 9123 4567")


class TestRe2Forms:
    def test_re2_form_has_no_lookaround(self) -> None:
        # RE2 (Google Cloud DLP custom info types) rejects lookaround. Every RE2 source must
        # be lookaround-free, or the managed profile fails INVALID_ARGUMENT on every call.
        lookaround = re.compile(r"\(\?[=!<]")
        for rows in p.NATIONAL_ID_PATTERNS.values():
            for info_type, pattern, _v in rows:
                re2 = p.re2_pattern_for(info_type, pattern)
                assert not lookaround.search(re2), f"{info_type} RE2 form has lookaround"

    def test_jp_row_has_a_distinct_re2_override(self) -> None:
        _info, pattern, _v = _rows_matching("JP_MY_NUMBER", "JP")[0]
        # The Python row uses lookarounds; the RE2 form must differ and be compilable.
        re2 = p.re2_pattern_for("JP_MY_NUMBER", pattern)
        assert re2 != pattern.pattern
        assert re.compile(re2).search("1234 5678 9018") is not None
