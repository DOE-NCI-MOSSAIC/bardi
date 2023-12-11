import os
import shutil
import unittest
from pathlib import Path

import pandas as pd

from bardi.data import data_handlers
from bardi.nlp_engineering import (CPUEmbeddingGenerator, CPULabelProcessor,
                                   CPUNormalizer, CPUPostProcessor,
                                   CPUPreTokenizer, CPUSplitter, NewSplit)
from bardi.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    """Tests the functionality of the Pipeline class"""

    def setUp(self):
        test_data_dir = os.path.join(Path().resolve(), 'tests', 'test_data')
        self.data_path = (f'{test_data_dir}/'
                          f'pipeline_test_df.pkl')
        self.write_dir = (f'{test_data_dir}/outputs')

        self.data_df = pd.read_pickle(self.data_path)
        # the fake dataset has the following fields:
        # id - unique id
        # state - one of "NY", "NH", "CA", "FL", "NM", "NC", "ID"]
        # letter - one of  ["A", "B", "C", "D"]
        # feature_1 integer from 01 to 5
        # feature_2 float
        # feature_3 boolen
        # text_1, text_2 and text_3 strings of words
        self.dataset = data_handlers.from_pandas(self.data_df)

    def test_adding_steps(self):
        """Test that steps are added to the pipeline as expected"""
        self.pipeline = Pipeline(dataset=self.dataset,
                                 write_path=self.write_dir,
                                 write_outputs='pipeline-outputs')

        # add pipeline steps
        fields = ['text_1', 'text_2', 'text_3']

        self.pipeline.add_step(CPUNormalizer(fields=fields))
        self.pipeline.add_step(CPUPreTokenizer(fields=fields))
        self.pipeline.add_step(CPUEmbeddingGenerator(fields=fields))

        # check the number of steps
        correct_ans = 3
        self.assertEqual(self.pipeline.num_steps, correct_ans,
                         ('Pipeline\'s num_steps attribute returns '
                          'an incorrect value'))

    def test_global_write_config_applied(self):
        """Test that applying a non-default global write config will
        apply to the writing of data from each module"""
        # supply non-default write config
        write_config = {"data_format": 'csv',
                        "data_format_args": {}}
        # loop through each step checking step write config
        self.pipeline = Pipeline(dataset=self.dataset,
                                 write_path=self.write_dir,
                                 write_outputs='debug',
                                 data_write_config=write_config)

        # add pipeline steps
        fields = ['text_1', 'text_2', 'text_3']
        new_field_name = 'text'

        self.pipeline.add_step(CPUNormalizer(fields=fields))
        self.pipeline.add_step(CPUPreTokenizer(fields=fields))
        self.pipeline.add_step(CPUEmbeddingGenerator(fields=fields))
        self.pipeline.add_step(CPUPostProcessor(fields=fields,
                                                field_rename=new_field_name,
                                                concat_fields=True))
        self.pipeline.add_step(CPULabelProcessor(
            fields=['state', 'feature_1']
        ))
        self.pipeline.add_step(CPUSplitter(NewSplit(
            split_proportions={'train': 0.7,
                               'test': 0.15,
                               'val': 0.15},
            unique_record_cols=['id'],
            group_cols=['id'],
            label_cols=None,
            random_seed=42)))

        # prepare test directory
        if os.path.exists(self.write_dir):
            shutil.rmtree(self.write_dir)
        os.makedirs(self.write_dir)

        # test that pipeline runs without failure
        self.pipeline.run_pipeline()

        # test that expected file outputs exist
        test_data_contents = os.listdir(self.write_dir)

        expected_files = ['id_to_token.json',
                          'id_to_label.json',
                          'embedding_matrix.npy',
                          'CPUNormalizerData.csv',
                          'CPUPreTokenizerData.csv',
                          'CPUEmbeddingGeneratorData.csv',
                          'CPUPostProcessorData.csv',
                          'CPULabelProcessorData.csv',
                          'CPUSplitterData.csv']
        self.assertTrue(
            set(expected_files).issubset(set(test_data_contents))
        )

    def test_pipeline_run(self):
        self.pipeline = Pipeline(dataset=self.dataset,
                                 write_path=self.write_dir,
                                 write_outputs='pipeline-outputs')

        # add pipeline steps
        fields = ['text_1', 'text_2', 'text_3']
        new_field_name = 'text'

        self.pipeline.add_step(CPUNormalizer(fields=fields))
        self.pipeline.add_step(CPUPreTokenizer(fields=fields))
        self.pipeline.add_step(CPUEmbeddingGenerator(fields=fields))
        self.pipeline.add_step(CPUPostProcessor(fields=fields,
                                                field_rename=new_field_name,
                                                concat_fields=True))
        self.pipeline.add_step(CPULabelProcessor(
            fields=['state', 'feature_1']
        ))
        self.pipeline.add_step(CPUSplitter(NewSplit(
            split_proportions={'train': 0.7,
                               'test': 0.15,
                               'val': 0.15},
            unique_record_cols=['id'],
            group_cols=['id'],
            label_cols=None,
            random_seed=42)))

        # prepare test directory
        if os.path.exists(self.write_dir):
            shutil.rmtree(self.write_dir)
        os.makedirs(self.write_dir)

        # test that pipeline runs without failure
        self.pipeline.run_pipeline()

        # test that expected file outputs exist
        test_data_contents = os.listdir(self.write_dir)
        expected_files = ['id_to_token.json',
                          'id_to_label.json',
                          'embedding_matrix.npy',
                          'bardi_processed_data.parquet']
        self.assertTrue(
            set(expected_files).issubset(set(test_data_contents))
        )

        # clean up
        if os.path.exists(self.write_dir):
            shutil.rmtree(self.write_dir)


    def test_getting_parameters(self):
        fields = ['text_1', 'text_2', 'text_3']
        new_field_name = 'text'

        # create pipeline
        self.pipeline = Pipeline(dataset=self.dataset,
                                 write_path=self.write_dir,
                                 write_outputs=False)

        # add pipeline steps
        self.pipeline.add_step(CPUNormalizer(fields=fields))
        self.pipeline.add_step(CPUPreTokenizer(fields=fields))
        self.pipeline.add_step(CPUEmbeddingGenerator(fields=fields))
        self.pipeline.add_step(CPUPostProcessor(fields=fields,
                                                field_rename=new_field_name,
                                                concat_fields=True))
        self.pipeline.add_step(CPULabelProcessor(
            fields=['state', 'feature_1']
        ))
        self.pipeline.add_step(CPUSplitter(NewSplit(
            split_proportions={'train': 0.7,
                               'test': 0.15,
                               'val': 0.15},
            unique_record_cols=['id'],
            group_cols=['id'],
            label_cols=None,
            random_seed=42)))

        # run pipeline
        self.pipeline.run_pipeline()

        # test condensed params
        pipeline_params = self.pipeline.get_parameters(condensed=True)

if __name__ == '__main__':
    unittest.main()
