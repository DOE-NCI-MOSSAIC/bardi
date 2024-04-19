import os
import re
import shutil
import unittest
from pathlib import Path

import polars as pl
from polars.testing import assert_series_equal, assert_series_not_equal

from bardi.nlp_engineering import CPUPreTokenizer


class TestPreTokenizer(unittest.TestCase):
    """Tests the functionality of the functions in bardi.nlp_engineering
    PreTokenizer class.
    """

    def setUp(self):

        # Mock polars DataFrame to test the correct behavior.
        self.df = pl.DataFrame(
            {
                "text_1": "this is a car. the car is blue. i do not like blue cars.",
                "text_2": "these programmers wrote this program. The programmer likes it.",
            }
        )
        self.pretokenizer = CPUPreTokenizer(fields=["text_1", "text_2"])
        self.retain_pretokenizer = CPUPreTokenizer(
            fields=["text_1", "text_2"], retain_input_fields=True
        )

    def test_pre_tokenizer(self):
        """A test to ensure the pre tokenizer splits strings
        correctly
        """

        data, artifacts = self.pretokenizer.run(self.df.to_arrow(), None)
        data = data.to_pandas()
        result = list(data["text_1"][0])
        answer = [
            "this",
            "is",
            "a",
            "car.",
            "the",
            "car",
            "is",
            "blue.",
            "i",
            "do",
            "not",
            "like",
            "blue",
            "cars.",
        ]
        self.assertEqual(result, answer, ("Incorrect output from the pre tokenizer"))

    def test_column_retention(self):
        """Do a full test of the CPUPreTokenizer class with column retention
        testing that the columns were retained and renamed as expected
        """

        data, artifacts = self.retain_pretokenizer.run(self.df.to_arrow(), None)

        df = pl.from_arrow(data)
        actual_cols = df.columns
        expected_cols = [
            "CPUPreTokenizer_input__text_1",
            "CPUPreTokenizer_input__text_2",
            "text_1",
            "text_2",
        ]

        # Test that all of the columns are there
        for col in actual_cols:
            self.assertIn(col, expected_cols)

        # Test for expected column contents
        for col in self.df.columns:
            actual_retained_series = df.get_column(f'CPUPreTokenizer_input__{col}')
            original_series = self.df.get_column(col)
            split_series = df.get_column(col)

            # retained series content should match original content
            assert_series_equal(actual_retained_series, original_series, check_names=False)

            # the new data in the column should not equal the original data (it was split!)
            assert_series_not_equal(split_series, original_series, check_names=False)

    def test_write_data(self):
        """A test to ensure that the pre-tokenizer's write function
        correctly produces a file as desired
        """
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

            self.pretokenizer.set_write_config(write_config)
            data, artifacts = self.pretokenizer.run(data=self.df.to_arrow(), artifacts=None)
            self.pretokenizer.write_data(write_path=test_data_dir, data=data)
            test_data_contents = os.listdir(test_data_dir)
            result = False
            for test_file in test_data_contents:
                test_file_match = re.search(f"CPUPreTokenizerData.{test_format}", test_file)
                if test_file_match:
                    result = True
                    written_test_file_path = os.path.join(test_data_dir, test_file_match.string)
                    os.remove(written_test_file_path)
            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
