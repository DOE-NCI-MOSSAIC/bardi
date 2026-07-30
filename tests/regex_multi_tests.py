"""Applies the regex library rule-by-rule to a real ORNL pathology report
sample, printing each intermediate result (visible via ``pytest -s``), and
asserts invariants on the final text.

The sample fixture is not distributed with the repository. The test is
skipped unless the pickle exists at the default location
(``tests/test_data/recurrence_raw_data_sample.pkl``) or at the path given in
the ``BARDI_RECURRENCE_SAMPLE`` environment variable. Row selection is
deterministic; override with ``BARDI_SAMPLE_ROW_SEED`` to inspect other rows.
"""

import os
import random
import re
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa

from bardi import nlp_engineering as nlp
from bardi.nlp_engineering import CPUNormalizer, PathologyReportRegexSet

from tests.regex_vectors import KNOWN_TOKENS

RECURRENCE_SAMPLE_PATH = os.environ.get(
    "BARDI_RECURRENCE_SAMPLE",
    str(Path(__file__).parent / "test_data" / "recurrence_raw_data_sample.pkl"),
)
RECURRENCE_SAMPLE_AVAILABLE = os.path.isfile(RECURRENCE_SAMPLE_PATH)
requires_recurrence_sample = unittest.skipUnless(
    RECURRENCE_SAMPLE_AVAILABLE,
    "recurrence raw data sample unavailable; set BARDI_RECURRENCE_SAMPLE",
)

SAMPLE_ROW_SEED = int(os.environ.get("BARDI_SAMPLE_ROW_SEED", "0"))


@requires_recurrence_sample
class TestRegexMultipleExpressions(unittest.TestCase):
    """Tests the regex library by applying the regex method
    in sequence and printing the results to screen."""

    def setUp(self) -> None:
        """Load the sample pickle and select the row under test.

        Reads the DataFrame from ``RECURRENCE_SAMPLE_PATH`` (the class is
        skipped when the file is absent, so this only runs when it exists),
        picks a deterministic row index from ``SAMPLE_ROW_SEED``, and sets
        the per-rule enable flags mirroring ``PathologyReportRegexSet``'s
        constructor arguments.
        """
        # Get data
        self.data = pd.read_pickle(RECURRENCE_SAMPLE_PATH)
        self.data.reset_index(inplace=True)
        self.max_int = self.data.shape[0]

        # Deterministic row selection (seed overridable via env var)
        self.row = random.Random(SAMPLE_ROW_SEED).randrange(self.max_int)

        self.lowercase = True
        self.handle_whitespaces = True  # 1
        self.remove_urls = True  # 2
        self.remove_special_punct = True  # 3
        self.remove_multiple_punct = True  # 4
        self.handle_angle_brackets = True  # 5

        self.replace_percent_sign = True  # 6
        self.handle_leading_digit_punct = True  # 7
        self.remove_leading_punct = True  # 8
        self.remove_trailing_punct = True  # 9
        self.handle_words_with_punct_spacing = True  # 10

        self.handle_math_spacing = True  # 11
        self.handle_dimension_spacing = True  # 12
        self.handle_measure_spacing = True  # 13
        self.handle_cassettes_spacing = True  # 14
        self.handle_dash_digits_spacing = True  # 15

        self.handle_literals_floats_spacing = True  # 16
        self.fix_pluralization = True  # 17
        self.handle_digits_words_spacing = True  # 18
        self.remove_phone_numbers = True  # 19
        self.remove_dates = True  # 20

        self.remove_time = True  # 21
        self.remove_addresses = True  # 22
        self.remove_dimensions = True  # 23
        self.remove_specimen = True  # 24
        self.remove_decimal_seg_numbers = True  # 25
        self.remove_large_digits_seq = True  # 26
        self.remove_large_floats_seq = True  # 27
        self.trunc_decimals = True  # 28
        self.remove_cassette_names = True  # 29

    def assert_normalized_invariants(self, text: str) -> None:
        """Assert the invariants any fully normalized report must satisfy.

        Checks that ``text`` is non-null, contains no ``\\r``/``\\n``/``\\t``
        (removed by rule 1), no consecutive whitespace (collapsed by the
        unconditional final spaces rule), no backslashes (removed by rules
        0/3), and no uppercase run other than the known ``*TOKEN``
        substitution strings (input is lowercased before the rules run).

        Args:
            text: The fully normalized report text to check.
        """
        self.assertIsNotNone(text)
        for escape_char in ("\r", "\n", "\t"):
            self.assertNotIn(escape_char, text)
        self.assertIsNone(
            re.search(r"\s\s", text), "found consecutive whitespace in normalized text"
        )
        self.assertNotIn("\\", text)
        for uppercase_run in re.findall(r"[A-Z]+", text):
            self.assertIn(
                uppercase_run,
                KNOWN_TOKENS,
                "unexpected uppercase run in normalized text",
            )

    def test_single(self) -> None:
        """Applies each enabled regex rule in library order to one report.

        Lowercases the selected row's ``text_all``, applies the rules with
        Python ``re.sub`` one at a time, and prints the pattern and
        intermediate text after every rule (run with ``pytest -s`` to
        inspect them — the stepwise trace is this test's purpose). Finally
        asserts the normalization invariants on the resulting text.

        Note:
            This chain intentionally uses Python's ``re`` engine (like the
            frozen ``tests/regex_tests.py``); the production polars engine
            is exercised by ``test_full_chain_normalizer``.
        """
        test_text = self.data["text_all"][self.row].lower()

        regex_sub_pair = nlp.get_escape_code_regex()
        pattern = regex_sub_pair["regex_str"]
        replacement = regex_sub_pair["sub_str"]
        print(f'\n Rule 0: Escape Codes -  pattern: {pattern}'
              'replacement: {replacement}\n')
        test_text = re.sub(pattern, replacement, test_text)
        original_text = test_text
        print(test_text)

        if self.handle_whitespaces:
            regex_sub_pair = nlp.get_whitespace_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 1 White Spaces: pattern: {pattern} replacement:'
                  f' {replacement} \n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_urls:
            regex_sub_pair = nlp.get_urls_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 2 URLs Replacement: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_special_punct:
            regex_sub_pair = nlp.get_special_punct_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 3 Chosen Punctuation Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_multiple_punct:
            regex_sub_pair = nlp.get_multiple_punct_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 4 Multiple Punctuation: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_angle_brackets:
            regex_sub_pair = nlp.get_angle_brackets_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 5 Angle Brackets Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.replace_percent_sign:
            regex_sub_pair = nlp.get_percent_sign_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 6 Replace Percent Sign: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_leading_digit_punct:
            regex_sub_pair = nlp.get_leading_digit_punctuation_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 7 Leading Digit Punctuation: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_leading_punct:
            regex_sub_pair = nlp.get_leading_punctuation_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 8 Leading Punctuation: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_trailing_punct:
            regex_sub_pair = nlp.get_trailing_punctuation_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 9 Trailing Punctuation: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_words_with_punct_spacing:
            regex_sub_pair = nlp.get_words_with_punct_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 10 Words with Punctuation: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_math_spacing:
            regex_sub_pair = nlp.get_math_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 11 Math Operator Spacing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_dimension_spacing:
            regex_sub_pair = nlp.get_dimension_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 12 Dimension spacing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_measure_spacing:
            regex_sub_pair = nlp.get_measure_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 13 Measure spacing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_cassettes_spacing:
            regex_sub_pair = nlp.get_cassettes_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 14 Special Specing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_dash_digits_spacing:
            regex_sub_pair = nlp.get_dash_digits_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 15 Dash Spacing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_literals_floats_spacing:
            regex_sub_pair = nlp.get_literals_floats_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 16 Literal Floats: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.fix_pluralization:
            regex_sub_pair = nlp.get_fix_pluralization_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 17 Plurals Attach: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.handle_digits_words_spacing:
            regex_sub_pair = nlp.get_digits_words_spacing_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 18 Digits Words Spacing: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_phone_numbers:
            regex_sub_pair = nlp.get_phone_number_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 19 Phone Number Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_dates:
            regex_sub_pair = nlp.get_dates_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 20 Dates Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_time:
            regex_sub_pair = nlp.get_time_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 21 Time Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_addresses:
            regex_sub_pair = nlp.get_address_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 22 Address Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_dimensions:
            regex_sub_pair = nlp.get_dimensions_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 23 Dimension Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_specimen:
            regex_sub_pair = nlp.get_specimen_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 24 Specimen Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_decimal_seg_numbers:
            regex_sub_pair = nlp.get_decimal_segmented_numbers_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 25 Decimal Segmented Numbers: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_large_digits_seq:
            regex_sub_pair = nlp.get_large_digits_seq_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 26 Large Digits: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_large_floats_seq:
            regex_sub_pair = nlp.get_large_float_seq_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 27 Large Floats: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.trunc_decimals:
            regex_sub_pair = nlp.get_trunc_decimals_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 28 Truncate Decimal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        if self.remove_cassette_names:
            regex_sub_pair = nlp.get_cassette_name_regex()
            pattern = regex_sub_pair["regex_str"]
            replacement = regex_sub_pair["sub_str"]
            print(f'\n Rule 29 Cassette Names Removal: pattern: {pattern}'
                  f'replacement: {replacement}\n')
            test_text = re.sub(pattern, replacement, test_text)
            print(test_text)

        regex_sub_pair = nlp.get_spaces_regex()
        pattern = regex_sub_pair["regex_str"]
        replacement = regex_sub_pair["sub_str"]
        print(f'\n Final Rule Additional Spaces: pattern: {pattern}'
              f'replacement: {replacement}\n')
        test_text = re.sub(pattern, replacement, test_text)
        print(test_text)

        print("********   ORIGINAL TEXT   ********")
        print(original_text)
        print("********   AFTER   ********")
        print(test_text)
        print(f'INDEX : {self.row}')

        self.assert_normalized_invariants(test_text)

    def test_full_chain_normalizer(self) -> None:
        """Runs the same row through the production CPUNormalizer chain.

        Normalizes the selected row with the full default
        ``PathologyReportRegexSet`` (``lowercase=True``) twice — building a
        fresh regex set and normalizer per run, since both mutate the
        substitution pairs in place — then asserts the normalization
        invariants on the output and that the two runs are identical.

        Note:
            The ``re.sub`` chain in ``test_single`` and the polars chain
            here may legitimately differ (different regex engines), so
            their outputs are not compared to each other.
        """
        raw_text = self.data["text_all"][self.row]
        table = pa.table({"text": [raw_text]})

        outputs = []
        for _ in range(2):
            # Fresh regex set and normalizer per run: both mutate the
            # substitution pairs in place.
            normalizer = CPUNormalizer(
                fields=["text"],
                regex_set=PathologyReportRegexSet().get_regex_set(),
                lowercase=True,
            )
            result, _ = normalizer.run(table)
            outputs.append(result.column("text").to_pylist()[0])

        print("********   FULL CHAIN (CPUNormalizer)   ********")
        print(outputs[0])
        print(f'INDEX : {self.row}')

        self.assert_normalized_invariants(outputs[0])
        self.assertEqual(outputs[0], outputs[1], "full chain is not deterministic")


if __name__ == '__main__':
    unittest.main()
