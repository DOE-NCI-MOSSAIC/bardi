import os
import re
import shutil
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa

from gaudi.data.data_handlers import from_pandas
from gaudi.nlp_engineering.splitter import CPUSplitter, NewSplit


class TestSplitter(unittest.TestCase):
    """Tests the functionality of the functions in gaudi.nlp_engineering
    Splitter class. The Splitter has two options MapSplit, NewSplit """

    def setUp(self):
        repo_path = Path().resolve()
        self.data_path = (f'{repo_path}/tests/test_data/'
                          f'split_test_df.pkl')
        self.data_df = pd.read_pickle(self.data_path)

        # Create a gaudi dataset object from a Pandas DataFrame
        self.data = from_pandas(df=self.data_df)

        self.splitter = CPUSplitter(NewSplit(
            split_proportions={'train': 0.7,
                               'test': 0.15,
                               'val': 0.15},
            unique_record_cols=['id'],
            group_cols=['state',
                        'letter'],
            label_cols=None,
            random_seed=42)
        )

    def test_new_splitter(self):
        """Set of tests to ensure that the data Splitter option:
        NewSplit function is correctly splitting the data into
        train/test an validation set based on the provided proportions."""

        split_data, artifacts = self.splitter.run(self.data.data)
        split_data_pd = split_data.to_pandas()

        # Compare whether the two columns have the same values.
        self.assertEqual(split_data_pd["split_correct"].tolist(),
                         split_data_pd["split"].tolist(),
                         'Incorrect split with NewSplitter.')

    def test_write_data(self):
        """A test to ensure that the splitter's write function
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

            self.splitter.set_write_config(write_config)
            data, artifacts = self.splitter.run(
                data=pa.Table.from_pandas(self.data_df),
                artifacts=None)
            self.splitter.write_data(write_path=test_data_dir, data=data)
            test_data_contents = os.listdir(test_data_dir)
            result = False
            for test_file in test_data_contents:
                test_file_match = re.search(
                    f'CPUSplitterData.{test_format}',
                    test_file)

                if test_file_match:
                    result = True
                    written_test_file_path = os.path.join(
                        test_data_dir,
                        test_file_match.string
                        )
                    os.remove(written_test_file_path)
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
