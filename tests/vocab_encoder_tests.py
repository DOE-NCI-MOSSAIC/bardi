import os
import re
import shutil
import unittest
from pathlib import Path

import polars as pl
import pyarrow as pa
from polars.testing import assert_series_equal

from bardi.nlp_engineering import CPUVocabEncoder


class TestVocabEncoder(unittest.TestCase):
    """Tests the functionality of the functions in bardi.nlp_engineering
    VocabEncoder class."""

    def setUp(self):

        # Mock polars DataFrame to test the correct behaviour.
        self.df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [["aa", "ee"], ["cc", "dd", "cc"], ["gg"]],
                "c": [["bb"], ["cc", "dd"], ["ee", "dd"]],
                "d": [["cc"], ["bb", "gg"], ["aa", "cc", "gg"]],
                "e": [["ee", "aa"], ["gg"], ["ff"]],
            }
        )
        self.id_to_token = {0: "aa", 1: "bb", 2: "cc", 3: "dd", 4: "ee", 5: "ff", 6: "gg"}

    def test_cpu_vocabencoder(self):
        """A test to ensure that the columns of a polars DataFrame
        (which are of type large_list) are correctly conatinated."""

        # ======== Test 4 fields without concatination ========
        fields = ["b", "c", "d", "e"]
        self.vocabencoder = CPUVocabEncoder(
            fields=fields, field_rename="text", concat_fields=False
        )
        data, artifacts = self.vocabencoder.run(
            data=self.df.to_arrow(), artifacts={"a": []}, id_to_token=self.id_to_token
        )
        data = data.select(fields)

        # Correct mapping
        correct_ans = {
            "b": [[0, 4], [2, 3, 2], [6]],
            "c": [[1], [2, 3], [4, 3]],
            "d": [[2], [1, 6], [0, 2, 6]],
            "e": [[4, 0], [6], [5]],
        }

        schema = pa.schema(
            [
                pa.field("b", pa.large_list(pa.int64())),
                pa.field("c", pa.large_list(pa.int64())),
                pa.field("d", pa.large_list(pa.int64())),
                pa.field("e", pa.large_list(pa.int64())),
            ]
        )

        correct_ans = pa.table(correct_ans, schema=schema)
        self.assertEqual(data, correct_ans, "Incorrect")

        # ======== Test 4 fields with concatination ========
        new_field_name = "text"
        self.vocabencoder = CPUVocabEncoder(
            fields=["b", "c", "d", "e"], field_rename=new_field_name, concat_fields=True
        )
        data, artifacts = self.vocabencoder.run(
            data=self.df.to_arrow(), artifacts={"a": []}, id_to_token=self.id_to_token
        )
        # check if the field was correctly renamed
        col_names = data.schema.names
        field_name_flag = True if new_field_name in col_names else False

        self.assertEqual(
            field_name_flag, True, ("Renaming the column" " after concat" " is incorrect.")
        )

        data = data.select([new_field_name])

        correct_ans = {
            new_field_name: [[0, 4, 1, 2, 4, 0], [2, 3, 2, 2, 3, 1, 6, 6], [6, 4, 3, 0, 2, 6, 5]]
        }

        schema = pa.schema([pa.field(new_field_name, pa.large_list(pa.int64()))])
        correct_ans = pa.table(correct_ans, schema=schema)
        self.assertEqual(data, correct_ans, "Incorrect")

        # ======== Test 2 fields without concatination ========
        fields = ["b", "c"]
        new_field_name = "text"

        self.vocabencoder = CPUVocabEncoder(
            fields=fields, field_rename=new_field_name, concat_fields=True
        )
        data, artifacts = self.vocabencoder.run(
            data=self.df.to_arrow(), artifacts={"a": []}, id_to_token=self.id_to_token
        )

        data = data.select([new_field_name])

        correct_ans = {new_field_name: [[0, 4, 1], [2, 3, 2, 2, 3], [6, 4, 3]]}

        schema = pa.schema([pa.field(new_field_name, pa.large_list(pa.int64()))])
        correct_ans = pa.table(correct_ans, schema=schema)
        self.assertEqual(data, correct_ans, "Incorrect")

    def test_column_retention(self):

        fields = ["b", "c", "d", "e"]
        self.vocabencoder = CPUVocabEncoder(
            fields=fields, field_rename="text", concat_fields=True, retain_input_fields=True
        )
        data, _ = self.vocabencoder.run(
            data=self.df.to_arrow(), artifacts={"a": []}, id_to_token=self.id_to_token
        )

        df = pl.from_arrow(data)
        actual_cols = df.columns
        expected_cols = ["a", "text"]
        expected_cols.extend([f"CPUVocabEncoder_input__{col}" for col in fields])

        # Test that all of the columns are there
        for col in actual_cols:
            self.assertIn(col, expected_cols)

        # Test for expected column contents
        for col in fields:
            actual_retained_series = df.get_column(f'CPUVocabEncoder_input__{col}')
            original_series = self.df.get_column(col)

            # retained series content should match original content
            assert_series_equal(actual_retained_series, original_series, check_names=False)

    def test_write_data(self):
        """A test to ensure that the vocab encoder's write function
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

            self.vocabencoder = CPUVocabEncoder(
                fields=["b", "c", "d", "e"], field_rename="text", concat_fields=False
            )
            self.vocabencoder.set_write_config(write_config)
            data, artifacts = self.vocabencoder.run(
                data=self.df.to_arrow(), artifacts={"a": []}, id_to_token=self.id_to_token
            )
            self.vocabencoder.write_data(write_path=test_data_dir, data=data)

            test_data_contents = os.listdir(test_data_dir)
            result = False
            for test_file in test_data_contents:
                test_file_match = re.search(f"CPUVocabEncoderData.{test_format}", test_file)
                if test_file_match:
                    result = True
                    written_test_file_path = os.path.join(test_data_dir, test_file_match.string)
                    os.remove(written_test_file_path)
            self.assertTrue(
                result,
                "The vocab encoder is not correctly " f"writing files for {test_format} format.",
            )


if __name__ == "__main__":
    unittest.main()
