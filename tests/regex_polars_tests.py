"""Characterization tests for the regex library under the production engine.

``tests/regex_tests.py`` (frozen) proves each pattern's behavior under Python's
``re.sub``, but production (``CPUNormalizer.run``) applies the patterns through
polars' Rust regex engine, whose semantics differ (``$N`` vs ``\\N``
backreferences, unicode classes, unset-group handling). These tests drive the
exact same vectors — plus new edge vectors — through the production path so
that any engine divergence or accidental pattern change is caught.

Expected values are characterization data captured from the production engine
(see tests/regex_vectors.py); they document current behavior rather than
blessing it as correct.
"""

import unittest
from typing import List, Optional

import pyarrow as pa

from bardi import nlp_engineering as nlp
from bardi.nlp_engineering import CPUNormalizer, PathologyReportRegexSet
from bardi.nlp_engineering.regex_library.regex_set import RegexSubPair

from tests.regex_vectors import EDGE_VECTORS, PARITY_VECTORS, SAME


def apply_polars(pair: RegexSubPair, texts: List[Optional[str]]) -> List[Optional[str]]:
    """Apply a single regex sub pair to texts via the production path.

    A fresh ``CPUNormalizer`` is built per call, and the pair is copied,
    because both ``CPUNormalizer.__init__`` and ``RegexSet.get_regex_set``
    mutate sub_strs in place.
    """
    normalizer = CPUNormalizer(fields=["text"], regex_set=[dict(pair)], lowercase=False)
    table = pa.table({"text": pa.array(texts, type=pa.string())})
    result, _ = normalizer.run(table)
    return result.column("text").to_pylist()


class TestRegexPolarsParity(unittest.TestCase):
    """Runs every vector from the frozen re-engine tests, plus new edge
    vectors, through the production polars engine."""

    def _check_getter(self, getter_name: str) -> None:
        """Batch all of a getter's vectors into one table and compare."""
        pair = getattr(nlp, getter_name)()
        parity = PARITY_VECTORS[getter_name]
        edge = EDGE_VECTORS.get(getter_name, [])

        inputs = [v["input"] for v in parity] + [v["input"] for v in edge]
        expected = [
            v["re_expected"] if v["polars_expected"] is SAME else v["polars_expected"]
            for v in parity
        ] + [v["polars_expected"] for v in edge]

        actual = apply_polars(pair, inputs)

        for text, want, got in zip(inputs, expected, actual):
            with self.subTest(getter=getter_name, input=text):
                self.assertEqual(
                    got,
                    want,
                    f"polars-engine output changed for {getter_name}",
                )

    # 0
    def test_escape_code_regex(self):
        self._check_getter("get_escape_code_regex")

    # 1
    def test_whitespace_regex(self):
        self._check_getter("get_whitespace_regex")

    # 2
    def test_urls_regex(self):
        self._check_getter("get_urls_regex")

    # 3
    def test_special_punct_regex(self):
        self._check_getter("get_special_punct_regex")

    # 4
    def test_multiple_punct_regex(self):
        self._check_getter("get_multiple_punct_regex")

    # 5
    def test_angle_brackets_regex(self):
        self._check_getter("get_angle_brackets_regex")

    # 6
    def test_percent_sign_regex(self):
        self._check_getter("get_percent_sign_regex")

    # 7
    def test_leading_digit_punctuation_regex(self):
        self._check_getter("get_leading_digit_punctuation_regex")

    # 8
    def test_leading_punctuation_regex(self):
        self._check_getter("get_leading_punctuation_regex")

    # 9
    def test_trailing_punctuation_regex(self):
        self._check_getter("get_trailing_punctuation_regex")

    # 10
    def test_words_with_punct_spacing_regex(self):
        self._check_getter("get_words_with_punct_spacing_regex")

    # 11
    def test_math_spacing_regex(self):
        self._check_getter("get_math_spacing_regex")

    # 12
    def test_dimension_spacing_regex(self):
        self._check_getter("get_dimension_spacing_regex")

    # 13
    def test_measure_spacing_regex(self):
        self._check_getter("get_measure_spacing_regex")

    # 14
    def test_cassettes_spacing_regex(self):
        self._check_getter("get_cassettes_spacing_regex")

    # 15
    def test_dash_digits_spacing_regex(self):
        self._check_getter("get_dash_digits_spacing_regex")

    # 16
    def test_literals_floats_spacing_regex(self):
        self._check_getter("get_literals_floats_spacing_regex")

    # 17
    def test_fix_pluralization_regex(self):
        self._check_getter("get_fix_pluralization_regex")

    # 18
    def test_digits_words_spacing_regex(self):
        self._check_getter("get_digits_words_spacing_regex")

    # 19
    def test_phone_number_regex(self):
        self._check_getter("get_phone_number_regex")

    # 20
    def test_dates_regex(self):
        self._check_getter("get_dates_regex")

    # 21
    def test_time_regex(self):
        self._check_getter("get_time_regex")

    # 22
    def test_address_regex(self):
        self._check_getter("get_address_regex")

    # 23
    def test_dimensions_regex(self):
        self._check_getter("get_dimensions_regex")

    # 24
    def test_specimen_regex(self):
        self._check_getter("get_specimen_regex")

    # 25
    def test_decimal_segmented_numbers_regex(self):
        self._check_getter("get_decimal_segmented_numbers_regex")

    # 26
    def test_large_digits_seq_regex(self):
        self._check_getter("get_large_digits_seq_regex")

    # 27
    def test_large_float_seq_regex(self):
        self._check_getter("get_large_float_seq_regex")

    # 28
    def test_trunc_decimals_regex(self):
        self._check_getter("get_trunc_decimals_regex")

    # 29
    def test_cassette_name_regex(self):
        self._check_getter("get_cassette_name_regex")

    # 30
    def test_duration_regex(self):
        self._check_getter("get_duration_regex")

    # 31
    def test_letter_num_seq_regex(self):
        self._check_getter("get_letter_num_seq_regex")

    # LAST
    def test_spaces_regex(self):
        self._check_getter("get_spaces_regex")

    def test_all_getters_covered(self):
        """Every getter with vectors has a dedicated test method above."""
        excluded = {"test_all_getters_covered", "test_null_passthrough"}
        tested = {
            name.replace("test_", "get_", 1)
            for name in dir(self)
            if name.startswith("test_") and name not in excluded
        }
        self.assertEqual(set(PARITY_VECTORS), tested)
        self.assertTrue(set(EDGE_VECTORS) <= tested)

    def test_null_passthrough(self):
        """Nulls survive every single-pattern substitution unchanged."""
        for getter_name in PARITY_VECTORS:
            with self.subTest(getter=getter_name):
                pair = getattr(nlp, getter_name)()
                self.assertEqual(apply_polars(pair, [None]), [None])


class TestRegexPolarsFullChain(unittest.TestCase):
    """Smoke test of the full default PathologyReportRegexSet chain through
    the production CPUNormalizer (style of normalizer_tests.py)."""

    def _run_default_chain(self, texts: List[Optional[str]]) -> List[Optional[str]]:
        # Fresh set per run: get_regex_set() and CPUNormalizer.__init__ both
        # mutate the pairs in place.
        normalizer = CPUNormalizer(
            fields=["text"],
            regex_set=PathologyReportRegexSet().get_regex_set(),
            lowercase=True,
        )
        table = pa.table({"text": pa.array(texts, type=pa.string())})
        result, _ = normalizer.run(table)
        return result.column("text").to_pylist()

    def test_default_chain_characterization(self):
        texts = [
            "Specimen received 03/10/01 at 3:34 pm.\nGross: measuring 1.3x0.7x0.1 cm; "
            "\\x0d block: 1-e >95% involved. Call (123) 456 7890.",
            "",
            None,
            "melanoma of the right arm",
        ]
        expected = [
            # Note: "block: 1-e" ends as "block 1 e", not CASSETTETOKEN —
            # rule 7 (leading digit punct) rewrites "1-e" to "1 e" before
            # rule 29 (cassette names) ever sees it. Order dependence.
            "specimen received DATETOKEN gross measuring DIMENSIONTOKEN cm "
            "block 1 e > 95 percent involved call PHONENUMTOKEN .",
            "",
            None,
            "melanoma of the right arm",
        ]
        self.assertEqual(self._run_default_chain(texts), expected)

    def test_default_chain_deterministic(self):
        texts = ["Specimen 03/10/01: 1.3x0.7 cm, block 7a f8; >95%."]
        self.assertEqual(self._run_default_chain(texts), self._run_default_chain(texts))


if __name__ == "__main__":
    unittest.main()
