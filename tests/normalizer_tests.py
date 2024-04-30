import os
import re
import shutil
import unittest
from pathlib import Path

import polars as pl
from polars.testing import assert_series_equal, assert_series_not_equal

from bardi.nlp_engineering import CPUNormalizer, PathologyReportRegexSet


class TestNormalizer(unittest.TestCase):
    """Tests the normalizer's methods and the implementation
    of regex substitutions. The individual regex functionalities
    themselves are tested in a different test class."""

    def setUp(self):
        """Execute the common setup code needed
        for all of the normalizer tests"""
        # Mock polars DataFrame to test the correct behaviour.
        self.df = pl.DataFrame(
            {
                "text_1": "At 1234 north 500 west provo ca 12345.\n The speciment:"
                "sh-22-0011300 3.89 x4.56cm. Or 4.3 km. ",
                "text_2": "Call  123 456 7890 .This is: 0.8943",
            }
        )
        self.path_regex_set = PathologyReportRegexSet(
            handle_whitespaces=True,
            remove_special_punct=True,
            handle_angle_brackets=True,
            replace_percent_sign=True,
            remove_phone_numbers=True,
            remove_dates=True,
            remove_addresses=True,
        )

        self.regex_set = self.path_regex_set.get_regex_set()

        self.standard_normalizer = CPUNormalizer(
            fields=["text_1", "text_2"],
            regex_set=self.regex_set,
            lowercase=True,
        )

        self.retain_normalizer = CPUNormalizer(
            fields=["text_1", "text_2"],
            regex_set=self.regex_set,
            lowercase=True,
            retain_input_fields=True,
        )

    def test_cpu_normalizer(self):
        """Do a full test of the CPU normalizer class ensuring
        the overall output is what is expected"""

        data, artifacts = self.standard_normalizer.run(self.df.to_arrow(), None)

        # Compare expected output table to the test table that
        # has been processed with the normalizer
        result_1 = data.column("text_1").to_pandas()[0]
        correct_1 = "at ADDRESSTOKEN the speciment SPECIMENTOKEN DIMENSIONTOKEN cm or 4.3 km "
        self.assertEqual(
            result_1,
            correct_1,
            ("The overall result of" " the normalizer does not match" "the expected output."),
        )

        result_2 = data.column("text_2").to_pandas()[0]
        correct_2 = "call PHONENUMTOKEN this is 0.8943"
        self.assertEqual(
            result_2,
            correct_2,
            ("The overall result of" " the normalizer does not match" "the expected output."),
        )

    def test_lowercase_subsitution(self):
        """Test lowercase_substitution option for the regex set"""
        lowercase_set = self.path_regex_set.get_regex_set(lowercase_substitution=True)

        ans1 = {"regex_str": "\\b[a-z]\\d{6,10}[.\\s]*", "sub_str": " letterdigitstoken "}

        self.assertEqual(
            lowercase_set[31],
            ans1,
            ("The lowercase_substitution option does not return the expected output."),
        )

        ans2 = {"regex_str": " \\d{1,2}d\\d{6,9}[.\\s]*", "sub_str": " durationtoken "}

        self.assertEqual(
            lowercase_set[30],
            ans2,
            ("The lowercase_substitution option does not return the expected output."),
        )

    def test_no_subsitution(self):
        """Test no_substitution option for the regex set"""
        no_substitution_set = self.path_regex_set.get_regex_set(no_substitution=True)

        ans1 = {"regex_str": "\\b[a-z]\\d{6,10}[.\\s]*", "sub_str": " "}

        self.assertEqual(
            no_substitution_set[31],
            ans1,
            ("The no_substitution option does not return the expected output."),
        )

        ans2 = {"regex_str": " \\d{1,2}d\\d{6,9}[.\\s]*", "sub_str": " "}

        self.assertEqual(
            no_substitution_set[30],
            ans2,
            ("The no_substitution option does not return the expected output."),
        )

    def test_column_retention(self):
        """Do a full test of the CPU normalizer class with column retention
        testing that the columns were retained and renamed as expected"""

        data, artifacts = self.retain_normalizer.run(self.df.to_arrow(), None)

        df = pl.from_arrow(data)
        actual_cols = df.columns
        expected_cols = [
            "CPUNormalizer_input__text_1",
            "CPUNormalizer_input__text_2",
            "text_1",
            "text_2",
        ]

        # Test that all of the columns are there
        for col in actual_cols:
            self.assertIn(col, expected_cols)

        # Test for expected column contents
        for col in self.df.columns:
            actual_retained_series = df.get_column(f"CPUNormalizer_input__{col}")
            original_series = self.df.get_column(col)
            cleaned_series = df.get_column(col)

            # retained series content should match original content
            assert_series_equal(actual_retained_series, original_series, check_names=False)

            # the new data in the column should not equal the original data (it was cleaned!)
            assert_series_not_equal(cleaned_series, original_series, check_names=False)

    def test_write_data(self):
        """A test to ensure that the normalizer's write function
        correctly produces a file as desired"""
        test_data_dir = os.path.join(Path().resolve(), "tests", "test_data", "outputs")

        # prepare test directory
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        os.makedirs(test_data_dir)

        test_formats = ["csv", "parquet"]

        for test_format in test_formats:
            if test_format == "parquet":
                write_config = {
                    "data_format": "parquet",
                    "data_format_args": {"compression": "snappy", "use_dictionary": False},
                }
            elif test_format == "csv":
                write_config = {"data_format": "csv", "data_format_args": {}}

            self.standard_normalizer.set_write_config(write_config)
            data, artifacts = self.standard_normalizer.run(data=self.df.to_arrow(), artifacts=None)
            self.standard_normalizer.write_data(write_path=test_data_dir, data=data)

            test_data_contents = os.listdir(test_data_dir)
            result = False
            for test_file in test_data_contents:
                test_file_match = re.search(f"CPUNormalizerData.{test_format}", test_file)
                if test_file_match:
                    result = True
                    written_test_file_path = os.path.join(test_data_dir, test_file_match.string)
                    os.remove(written_test_file_path)
            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
