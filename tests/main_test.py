"""GAuDI test driver. Executes tests of all the modules.
Run with:
`python -m tests.main_test`
"""
import unittest

from tests.data_handlers_tests import TestDataHandlers
from tests.embedding_generator_tests import TestEmbeddingGenerator
from tests.label_processor_tests import TestLabelProcessor
from tests.normalizer_tests import TestNormalizer
from tests.pipeline_tests import TestPipeline
from tests.polars_utils_tests import TestPolarsUtils
from tests.pretokenizer_tests import TestPreTokenizer
from tests.regex_tests import TestRegexExpressions
from tests.splitter_tests import TestSplitter
from tests.tokenizer_tests import TestTokenizers
from tests.vocab_encoder_tests import TestVocabEncoder


def suite(module_name):
    """Build a test suite from module's TestCase"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(module_name)
    return suite


def main():
    runner = unittest.TextTestRunner()

    # Test Data Handler Module
    print("Test Data Handlers Module")
    data_handlers_test_suite = suite(TestDataHandlers)
    runner.run(data_handlers_test_suite)

    # Test Regular Expressions Library
    print("Test Regular Expressions' Library")
    regex_test_suite = suite(TestRegexExpressions)
    runner.run(regex_test_suite)

    # Test Polars Utils
    print("Test Polars Utils")
    polars_utils_test_suite = suite(TestPolarsUtils)
    runner.run(polars_utils_test_suite)

    # Test Normalizer Module
    print("Test Normalizer Module")
    normalizer_test_suite = suite(TestNormalizer)
    runner.run(normalizer_test_suite)

    # Test Tokenizer Module
    print("Test Tokenizer Module")
    tokenizer_test_suite = suite(TestTokenizers)
    runner.run(tokenizer_test_suite)

    # Test Pretokenizer Module
    print("Test Pretokenizer Module")
    pretokenizer_test_suite = suite(TestPreTokenizer)
    runner.run(pretokenizer_test_suite)

    # Test Generete Embeddings Module
    print("Test Embedding Generator Module")
    embedding_gen_test_suite = suite(TestEmbeddingGenerator)
    runner.run(embedding_gen_test_suite)

    # Test VocabEncoder Module
    print("Test VocabEncoder Module")
    vocabencoder_test_suite = suite(TestVocabEncoder)
    runner.run(vocabencoder_test_suite)

    # Test Splitter Module
    print("Test Splitter Module")
    splitter_test_suite = suite(TestSplitter)
    runner.run(splitter_test_suite)

    # Test Label Generetor
    print("Test Label Processor Module")
    label_processor_test_suite = suite(TestLabelProcessor)
    runner.run(label_processor_test_suite)

    # Test Pipeline
    print("Test Pipeline Module!")
    pipeline_test_suite = suite(TestPipeline)
    runner.run(pipeline_test_suite)


if __name__ == "__main__":
    main()
