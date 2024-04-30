import os
import re
import shutil
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa

from bardi.data import from_pandas
from bardi.nlp_engineering.embedding_generator import CPUEmbeddingGenerator


class TestEmbeddingGenerator(unittest.TestCase):
    """Tests the functionality of the functions in bardi.nlp_engineering
    Embedding Generetor class."""

    def setUp(self):
        repo_path = Path().resolve()
        self.data_path = f"{repo_path}/tests/test_data/" f"embed_gen_test_df.pkl"
        self.data_df = pd.read_pickle(self.data_path)

        # Create a bardi dataset object from a Pandas DataFrame
        self.data = from_pandas(df=self.data_df)
        self.word2vec_model_path = "word2vec.model"

        self.embedding_generator = CPUEmbeddingGenerator(
            fields=["text_1", "text_2", "text_3"],
            min_word_count=1,
            window=5,
            checkpoint_path=self.word2vec_model_path,
            load_saved_model=False,
        )

    def test_multiple_text_cols(self):
        """A test to ensure that multiple
        text columns are considered when creating word2vec
        vocab"""

        artifacts = {}
        data, artifacts = self.embedding_generator.run(data=self.data.data, artifacts=artifacts)

        # Compare whether the two columns have the same values.
        self.assertEqual(602, len(artifacts["id_to_token"]), "Incorrect.")

    def test_write_data(self):
        """A test to ensure that the embedding generator's write function
        correctly produces files as desired"""
        test_data_dir = os.path.join(Path().resolve(), "tests", "test_data", "outputs")

        # prepare test directory
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        os.makedirs(test_data_dir)

        # Test data writing
        test_formats = ["csv", "parquet"]
        for test_format in test_formats:
            if test_format == "parquet":
                write_config = {
                    "data_format": "parquet",
                    "data_format_args": {"compression": "snappy", "use_dictionary": False},
                }
            elif test_format == "csv":
                write_config = {"data_format": "csv", "data_format_args": {}}

            self.embedding_generator.set_write_config(write_config)
            data, artifacts = self.embedding_generator.run(
                data=pa.Table.from_pandas(self.data_df), artifacts={}
            )

            self.embedding_generator.write_data(write_path=test_data_dir, data=data)
            self.embedding_generator.write_artifacts(write_path=test_data_dir, artifacts=artifacts)
            test_data_contents = os.listdir(test_data_dir)
            result = False
            for test_file in test_data_contents:
                test_file_match = re.search(f"CPUEmbeddingGeneratorData.{test_format}", test_file)
                if test_file_match:
                    result = True
                    written_test_file_path = os.path.join(test_data_dir, test_file_match.string)
                    os.remove(written_test_file_path)
            self.assertTrue(
                result, "Data was not written correctly from the " "embedding generator."
            )

            embedding_matrix_path = os.path.join(test_data_dir, "embedding_matrix.npy")
            embedding_matrix_result = os.path.isfile(embedding_matrix_path)
            os.remove(embedding_matrix_path)
            self.assertTrue(embedding_matrix_result, "Embedding matrix was not written.")

            vocab_path = os.path.join(test_data_dir, "id_to_token.json")
            vocab_result = os.path.isfile(vocab_path)
            os.remove(vocab_path)
            self.assertTrue(vocab_result, "Vocab was not written.")

    def tearDown(self):
        os.remove(self.word2vec_model_path)


if __name__ == "__main__":
    unittest.main()
