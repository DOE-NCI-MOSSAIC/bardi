import os
import re
import shutil
import unittest
from pathlib import Path

import polars as pl

from bardi.nlp_engineering.label_processor import CPULabelProcessor


class TestLabelProcessor(unittest.TestCase):
    """Tests the functionality of the functions in bardi.nlp_engineering
    LabelProcessor class. """

    def setUp(self):

        # Mock polars DataFrame to test the correct behaviour.
        self.df = pl.DataFrame(
            {
                'task_1': ['A', 'B', 'B', 'A', 'C', 'A', 'C'],
                'task_2': [0, -1, 0, 1, 2, -1, 0],
                'task_3': [0, 1, 2, 0, 1, 2, None],
                'task_4': [[0], [1], [2], [2], [0], [1], [1]]
            }
        )

    def test_label_processor(self):
        """ A test to ensure that the ids are mapped to labels
        as expecteds."""

        # ======== Test string ids to labels ========
        label_processor = CPULabelProcessor(fields='task_1')
        data, artifacts = label_processor.run(self.df.to_arrow(), None)
        answer = {'0': 'A', '1': 'B', '2': 'C'}

        self.assertEqual(artifacts["id_to_label"]["task_1"], answer,
                         ('Incorrect output from the label processor.'))

        # ======== Test integer ids to labels ========
        label_processor = CPULabelProcessor(fields=['task_1', 'task_2'])
        data, artifacts = label_processor.run(self.df.to_arrow(), None)
        answer = {'0': '-1', '1': '0', '2': '1', '3': '2'}

        self.assertEqual(artifacts["id_to_label"]["task_2"], answer,
                         ('Incorrect output from the label processor.'))

        # ======== Test when None is inclued in the values  ========
        label_processor = CPULabelProcessor(fields='task_3')
        data, artifacts = label_processor.run(self.df.to_arrow(), None)

        answer = {'0': 'None', '1': '0', '2': '1', '3': '2'}

        self.assertEqual(artifacts["id_to_label"]["task_3"], answer,
                         ('Incorrect output from the label processor.'))

        # ======== Test when lists are included in the column ========
        # Need to re-evaluate this test. Upgrade in Polars might be handling
        # this differently
        # Now throws an error during the execution

        # label_processor = CPULabelProcessor(fields='task_4')
        # data, artifacts = label_processor.run(self.df.to_arrow(), None)

    def test_write_data(self):
        """A test to ensure that the label processor's write function
        correctly produces files as desired"""
        test_data_dir = os.path.join(Path().resolve(), 'tests', 'test_data', 'outputs')

        # prepare test directory
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        os.makedirs(test_data_dir)
    
        label_processor = CPULabelProcessor(fields=['task_1', 'task_2'])

        # Test data writing
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
            
            label_processor.set_write_config(write_config)
            data, artifacts = label_processor.run(
                data=self.df.to_arrow(), artifacts={}
            )
            label_processor.write_data(write_path=test_data_dir,
                                       data=data)
            label_processor.write_artifacts(write_path=test_data_dir,
                                            artifacts=artifacts)
            test_data_contents = os.listdir(test_data_dir)
            result = False
            for test_file in test_data_contents:
                test_file_match = re.search(f'CPULabelProcessorData.{test_format}', test_file)
                if test_file_match:
                    result = True
                    written_test_file_path = os.path.join(test_data_dir,
                                                          test_file_match.string)
                    os.remove(written_test_file_path)
            self.assertTrue(result, "Data was not written correctly from the "
                            "label processor.")

            id_to_label_path = os.path.join(test_data_dir, 'id_to_label.json')
            id_to_label_result = os.path.isfile(id_to_label_path)
            os.remove(id_to_label_path)
            self.assertTrue(id_to_label_result, "id_to_label was not written.")


if __name__ == '__main__':
    unittest.main()
