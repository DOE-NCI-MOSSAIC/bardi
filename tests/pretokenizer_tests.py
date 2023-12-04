import os
import re
import shutil
import unittest
from pathlib import Path

import polars as pl

from gaudi.nlp_engineering.pre_tokenizer import CPUPreTokenizer


class TestPreTokenizer(unittest.TestCase):
    """Tests the functionality of the functions in gaudi.nlp_engineering
    PreTokenizer class. """

    def setUp(self):

        # Mock polars DataFrame to test the correct behaviour.
        self.df = pl.DataFrame(
            {
                'text_1': 'this is a car. the car is blue. i do not like blue cars.',
                'text_2': 'these programmers wrote this program. The programmer likes it.',
            }
        )
        self.pretokenizer = CPUPreTokenizer(fields=['text_1', 'text_2'])

    def test_pre_tokenizer(self):
        """A test to ensure the pre tokenizer splits strings
        correctly"""

        data, artifacts = self.pretokenizer.run(self.df.to_arrow(), None)
        data = data.to_pandas()
        result = list(data["text_1"][0])
        answer = ['this', 'is', 'a', 'car.', 'the', 'car', 'is', 'blue.',
                  'i', 'do', 'not', 'like', 'blue', 'cars.']
        self.assertEqual(result, answer,
                         ('Incorrect output from the pre tokenizer'))

    def test_write_data(self):
        """A test to ensure that the pre-tokenizer's write function
        correctly produces a file as desired"""
        test_data_dir = os.path.join(Path().resolve(), 'tests',
                                     'test_data', 'outputs')

        # prepare test directory
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        os.makedirs(test_data_dir)

        test_formats = ['csv', 'parquet']

        for test_format in test_formats:
            if test_format == 'parquet':
                write_config = {"data_format": 'parquet',
                                "data_format_args":
                                    {"compression": 'snappy',
                                     "use_dictionary": False}}
            elif test_format == 'csv':
                write_config = {"data_format": 'csv',
                                "data_format_args": {}}

            self.pretokenizer.set_write_config(write_config) 
            data, artifacts = self.pretokenizer.run(data=self.df.to_arrow(),
                                                    artifacts=None)
            self.pretokenizer.write_data(write_path=test_data_dir,
                                         data=data)
            test_data_contents = os.listdir(test_data_dir)
            result = False
            for test_file in test_data_contents:
                test_file_match = re.search(f'CPUPreTokenizerData.{test_format}',
                                            test_file)
                if test_file_match:
                    result = True
                    written_test_file_path = os.path.join(
                        test_data_dir,
                        test_file_match.string)
                    os.remove(written_test_file_path)
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
