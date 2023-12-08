import os
import re
import shutil
import unittest
from pathlib import Path

import polars as pl
import pyarrow as pa

from bardi.nlp_engineering.post_processor import CPUPostProcessor


class TestPostProcessor(unittest.TestCase):
    """Tests the functionality of the functions in bardi.nlp_engineering
    PostProcessor class. """

    def setUp(self):

        # Mock polars DataFrame to test the correct behaviour.
        self.df = pl.DataFrame(
            {
                'a': [1, 2, 3],
                'b': [['aa', 'ee'], ['cc', 'dd', 'cc'], ['gg']],
                'c': [['bb'], ['cc', 'dd'], ['ee', 'dd']],
                'd': [['cc'], ['bb', 'gg'], ['aa', 'cc', 'gg']],
                'e': [['ee', 'aa'], ['gg'], ['ff']],
            }
        )
        self.id_to_token = {0: "aa", 1: "bb", 2: "cc", 3: "dd",
                            4: "ee", 5: "ff", 6: "gg"}

    def test_cpu_postprocessor(self):
        """ A test to ensure that the columns of a polars DataFrame
        (which are of type large_list) are correctly conatinated."""

        # ======== Test 4 fields without concatination ========
        fields = ['b', 'c', 'd', 'e']
        self.postprocessor = CPUPostProcessor(fields=fields,
                                              field_rename="text",
                                              concat_fields=False)
        data, artifacts = self.postprocessor.run(data=self.df.to_arrow(),
                                                 artifacts={'a': []},
                                                 id_to_token=self.id_to_token)
        data = data.select(fields)

        # Correct mapping
        correct_ans = {
                       "b": [[0, 4], [2, 3, 2], [6]],
                       "c": [[1], [2, 3], [4, 3]],
                       "d": [[2], [1, 6], [0, 2, 6]],
                       "e": [[4, 0], [6], [5]]
                       }

        schema = pa.schema([pa.field('b', pa.large_list(pa.int64())),
                            pa.field('c', pa.large_list(pa.int64())),
                            pa.field('d', pa.large_list(pa.int64())),
                            pa.field('e', pa.large_list(pa.int64()))])

        correct_ans = pa.table(correct_ans, schema=schema)
        self.assertEqual(data, correct_ans, "Incorrect")

        # ======== Test 4 fields with concatination ========
        new_field_name = "text"
        self.postprocessor = CPUPostProcessor(fields=['b', 'c', 'd', 'e'],
                                              field_rename=new_field_name,
                                              concat_fields=True)
        data, artifacts = self.postprocessor.run(data=self.df.to_arrow(),
                                                 artifacts={'a': []},
                                                 id_to_token=self.id_to_token)
        # check if the field was correctly renamed
        col_names = data.schema.names
        field_name_flag = True if new_field_name in col_names else False

        self.assertEqual(field_name_flag, True, ('Renaming the column'
                                                 ' after concat'
                                                 ' is incorrect.'))

        data = data.select([new_field_name])

        correct_ans = {
            new_field_name: [[0, 4, 1, 2, 4, 0],
                             [2, 3, 2, 2, 3, 1, 6, 6],
                             [6, 4, 3, 0, 2, 6, 5]]
        }

        schema = pa.schema([pa.field(new_field_name,
                                     pa.large_list(pa.int64())
                                     )
                            ])
        correct_ans = pa.table(correct_ans, schema=schema)
        self.assertEqual(data, correct_ans, "Incorrect")

        # ======== Test 2 fields without concatination ========
        fields = ['b', 'c']
        new_field_name = "text"

        self.postprocessor = CPUPostProcessor(fields=fields,
                                              field_rename=new_field_name,
                                              concat_fields=True)
        data, artifacts = self.postprocessor.run(data=self.df.to_arrow(),
                                                 artifacts={'a': []},
                                                 id_to_token=self.id_to_token)

        data = data.select([new_field_name])

        correct_ans = {
            new_field_name: [[0, 4, 1],
                             [2, 3, 2, 2, 3],
                             [6, 4, 3]]
        }

        schema = pa.schema([pa.field(new_field_name,
                                     pa.large_list(pa.int64())
                                     )
                            ])
        correct_ans = pa.table(correct_ans, schema=schema)
        self.assertEqual(data, correct_ans, "Incorrect")

    def test_write_data(self):
        """A test to ensure that the post processor's write function
        correctly produces a file as desired"""
        test_data_dir = os.path.join(Path().resolve(), 'tests', 'test_data', 'outputs')

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

            self.postprocessor = CPUPostProcessor(fields=['b', 'c', 'd', 'e'],
                                                  field_rename="text",
                                                  concat_fields=False)
            self.postprocessor.set_write_config(write_config)
            data, artifacts = self.postprocessor.run(
                data=self.df.to_arrow(),
                artifacts={'a': []},
                id_to_token=self.id_to_token)
            self.postprocessor.write_data(write_path=test_data_dir, data=data)

            test_data_contents = os.listdir(test_data_dir)
            result = False
            for test_file in test_data_contents:
                test_file_match = re.search(
                    f'CPUPostProcessorData.{test_format}', test_file)
                if test_file_match:
                    result = True
                    written_test_file_path = os.path.join(
                        test_data_dir, test_file_match.string)
                    os.remove(written_test_file_path)
            self.assertTrue(result, "The post processor is not correctly "
                            f"writing files for {test_format} format.")


if __name__ == '__main__':
    unittest.main()
