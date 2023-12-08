import os
import re
import shutil
import unittest
from pathlib import Path

import polars as pl

from bardi.nlp_engineering.normalizer import CPUNormalizer
from bardi.nlp_engineering.regex_library.pathology_report import PathologyReportRegexSet


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
                "text_1": "At 1234 north 500 west provo ca 12345.\n The speciment: sh-22-0011300 3.89 x4.56cm. Or 4.3 km. ",
                "text_2": "Call  123 456 7890 .This is: 0.8943",
            }
        )
        regex_set = PathologyReportRegexSet(
            handle_whitespaces=True,
            remove_special_punct=True,
            handle_angle_brackets=True,
            replace_percent_sign=True,
            remove_phone_numbers=True,
            remove_dates=True,
            remove_addresses=True,
        ).get_regex_set()
        self.normalizer = CPUNormalizer(
            fields=["text_1", "text_2"],
            regex_set=regex_set,
            lowercase=True,
        )

    def test_cpu_normalizer(self):
        """Do a full test of the CPU normalizer class ensuring
        the overall output is what is expected"""

        data, artifacts = self.normalizer.run(self.df.to_arrow(), None)

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

            self.normalizer.set_write_config(write_config)
            data, artifacts = self.normalizer.run(data=self.df.to_arrow(), artifacts=None)
            self.normalizer.write_data(write_path=test_data_dir, data=data)

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
