import unittest
from unittest import TestCase

import polars as pl
from polars.testing import assert_series_equal

from bardi import nlp_engineering as nlp

HF_SHARED_CACHE = "/mnt/nci/scratch/hf_shared_cache"


class TestTokenizers(TestCase):
    """Tests the correctness of the functions in bardi's tokenizers' module"""

    # SORT OF DONE - Test loading different tokenizers from hf cache
    # DONE - Test using tokenizer passed through artifacts
    # DONE - Test providing tokenizer params vs. not
    # DONE - Test tokenization output with concat fields
    # DONE - Test tokenization output with non-concat fields
    # DONE - Test column retention
    # DONE - Test concat text retention

    def setUp(self):
        self.hf_cache_dir = HF_SHARED_CACHE
        # Mock polars DataFrame to test the correct behavior.
        self.df = pl.DataFrame(
            {
                "text_1": "this is a car. the car is blue. i do not like blue cars.",
                "text_2": "these programmers wrote this program. The programmer likes it.",
            }
        )
        self.fields = ["text_1", "text_2"]

    def test_loading_hf_tokenizers(self):
        """Tests correctness of loading the supported tokenizers"""

        checkpoint_paths = [
            "Clinical-BigBird",
            "Clinical-Longformer",
            "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "huggingface_llama_2/Llama-2-7b-chat-hf",
            "google/flan-t5-base",
            "UFNLP/gatortron-base",
        ]

        max_length = 15
        sequences = [
            "This note is about this and that.",
            "The tokenizer object can handle the conversion to specific framework tensors.",
        ]

        # Confirm that the tokenizer is fast and that the correct model got loaded
        for checkpoint_path in checkpoint_paths:
            checkpoint_path = f"{self.hf_cache_dir}/{checkpoint_path}"
            tokenizer = nlp.load_hf_tokenizer(checkpoint_path)
            model_inputs = tokenizer(sequences, truncation=True, max_length=max_length)
            self.assertTrue(tokenizer.is_fast)
            self.assertIn(
                "input_ids", model_inputs.keys(), f"Incorrect tokenizer loading: {checkpoint_path}"
            )

    def test_apply_clinical_bigbird_each_field(self):

        # Set up and run the Tokenizer Encoder
        tokenizer_encoder = nlp.CPUTokenizerEncoder(
            fields=self.fields, model_name="Clinical-BigBird", hf_cache_dir=self.hf_cache_dir
        )
        data, _ = tokenizer_encoder.run(data=self.df.to_arrow())
        df = pl.from_arrow(data)

        # Set the expected output of the test data with this model
        expected_text_1 = [
            65,
            529,
            419,
            358,
            1198,
            114,
            363,
            1198,
            419,
            4272,
            114,
            1413,
            567,
            508,
            689,
            4272,
            5107,
            114,
            66,
        ]

        # Set the expected output of the test data with this model
        expected_text_2 = [
            65,
            878,
            24968,
            2731,
            529,
            1531,
            114,
            484,
            24393,
            7933,
            441,
            114,
            66,
        ]

        expected_text_1_input_ids = expected_text_1
        expected_text_2_input_ids = expected_text_2

        # Get the actual output
        actual_text_1_input_ids = df.get_column("text_1_input_ids")
        actual_text_2_input_ids = df.get_column("text_2_input_ids")

        # Check if they are the same
        assert_series_equal(
            actual_text_1_input_ids,
            pl.Series([expected_text_1_input_ids], dtype=pl.List(inner=pl.Int32())),
            check_names=False,
        )
        assert_series_equal(
            actual_text_2_input_ids,
            pl.Series([expected_text_2_input_ids], dtype=pl.List(inner=pl.Int32())),
            check_names=False,
        )

    def test_apply_clinical_bigbird_concat_fields(self):

        tokenizer_encoder = nlp.CPUTokenizerEncoder(
            fields=self.fields,
            model_name="Clinical-BigBird",
            hf_cache_dir=self.hf_cache_dir,
            concat_fields=True,
        )

        data, _ = tokenizer_encoder.run(data=self.df.to_arrow())

        df = pl.from_arrow(data)

        expected_text_input_ids = [
            65,
            529,
            419,
            358,
            1198,
            114,
            363,
            1198,
            419,
            4272,
            114,
            1413,
            567,
            508,
            689,
            4272,
            5107,
            114,
            878,
            24968,
            2731,
            529,
            1531,
            114,
            484,
            24393,
            7933,
            441,
            114,
            66,
        ]

        assert_series_equal(
            df.get_column("input_ids"),
            pl.Series([expected_text_input_ids], dtype=pl.List(pl.Int32())),
            check_names=False,
        )

    def test_apply_clinical_bigbird_retain_concat_fields(self):
        tokenizer_encoder = nlp.CPUTokenizerEncoder(
            fields=self.fields,
            model_name="Clinical-BigBird",
            hf_cache_dir=self.hf_cache_dir,
            concat_fields=True,
            retain_concat_field=True,
        )
        data, _ = tokenizer_encoder.run(data=self.df.to_arrow())
        df = pl.from_arrow(data)

        self.assertIn("text", df.columns)
        self.assertIn("input_ids", df.columns)
        self.assertIn("attention_mask", df.columns)

    def test_apply_clinical_bigbird_retain_input_fields(self):
        tokenizer_encoder = nlp.CPUTokenizerEncoder(
            fields=self.fields,
            model_name="Clinical-BigBird",
            hf_cache_dir=self.hf_cache_dir,
            concat_fields=True,
            retain_input_fields=True,
            retain_concat_field=True,
        )
        data, _ = tokenizer_encoder.run(data=self.df.to_arrow())
        df = pl.from_arrow(data)

        actual_cols = df.columns

        # Confirm all of the expected columns are present and contents are correct
        for field in self.fields:
            self.assertIn(f"CPUTokenizerEncoder_input__{field}", actual_cols)
            self.assertIn("text", actual_cols)
            self.assertIn("input_ids", actual_cols)
            self.assertIn("attention_mask", actual_cols)

            actual_retained_series = df.get_column(f"CPUTokenizerEncoder_input__{field}")
            original_series = self.df.get_column(field)

            # retained series content should match original content
            assert_series_equal(actual_retained_series, original_series, check_names=False)

    def test_loading_tokenizer_from_artifacts(self):
        model_name = f"{self.hf_cache_dir}/UFNLP/gatortron-base"
        tokenizer = nlp.load_hf_tokenizer(model_name)

        artifacts = {"tokenizer_model": tokenizer}

        tokenizer_encoder = nlp.CPUTokenizerEncoder(fields=self.fields, concat_fields=True)
        data, _ = tokenizer_encoder.run(data=self.df.to_arrow(), artifacts=artifacts)
        df = pl.from_arrow(data)

        self.assertIn("input_ids", df.columns)
        self.assertIn("token_type_ids", df.columns)
        self.assertIn("attention_mask", df.columns)

    def test_no_tokenizer_provided(self):
        tokenizer_encoder = nlp.CPUTokenizerEncoder(fields=self.fields, concat_fields=True)

        with self.assertRaises(AttributeError):
            tokenizer_encoder.run(data=self.df.to_arrow())

    def test_apply_tokenizer_w_params(self):
        tokenizer_encoder = nlp.CPUTokenizerEncoder(
            fields=self.fields,
            model_name="Clinical-BigBird",
            hf_cache_dir=self.hf_cache_dir,
            concat_fields=True,
            tokenizer_params={"model_max_length": 6}
        )

        data, _ = tokenizer_encoder.run(data=self.df.to_arrow())
        df = pl.from_arrow(data)

        expected_text_input_ids = [65, 529, 419, 358, 1198, 66]

        assert_series_equal(
            df.get_column("input_ids"),
            pl.Series([expected_text_input_ids], dtype=pl.List(pl.Int32())),
            check_names=False,
        )

    def test_default_setting_model_max_length(self):
        tokenizer_encoder = nlp.CPUTokenizerEncoder(
            fields=self.fields,
            model_name="huggingface_llama_2/Llama-2-7b-chat-hf",
            hf_cache_dir=self.hf_cache_dir,
            concat_fields=True
        )

        _, _ = tokenizer_encoder.run(data=self.df.to_arrow())

        self.assertEqual(tokenizer_encoder.tokenizer_model.model_max_length, 4096)


if __name__ == "__main__":
    unittest.main()
