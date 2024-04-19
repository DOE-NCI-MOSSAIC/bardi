"""Train a tokenizer for transformer-based models"""

import os
from abc import abstractmethod
from typing import List, Tuple, TypedDict, Union

import polars as pl
import pyarrow as pa
from tokenizers import Tokenizer

from bardi.nlp_engineering.utils import tokenizers_lib
from bardi.nlp_engineering.utils.tokenizers_lib import TrainableTokenizer
from bardi.nlp_engineering.utils.validations import validate_pyarrow_table, validate_str_cols
from bardi.pipeline import DataWriteConfig, Step


class TokenizerTrainerArtifactsWriteConfig(TypedDict):
    """Indicates the keys and data types expected in an
    artifacts write config dict for the embedding generator
    if overwriting the default configuration."""

    vocab_format: str
    vocab_format_args: dict


class TokenizerTrainer(Step):
    """The TokenizerTrainer class provides ability to train:

    1) A NEW TOKENIZER FORM AN OLD ONE
        Train a new tokenizer based on a provided tokenizer (from_old_flag).
        Provide a trained tokenizers associated with given architecture
        (BERT, LLAMA) and train a new tokenizer from scratch that
        is configurated to the provided architecture.

    2) TRAIN NEW ARCHITECTURE AGNOSTIC TOKENIZER
        Use one of the supported tokenizer algorithms to train a new
        tokenizer from scratch.

    Note
    ----

    Avoid the direct instantiation of the TokenizerTrainer class
    and instead instantiate one of the child classes depending
    on hardware configuration.

    Attributes
    ----------

    field : str
        the name of the column containing text
    tokenizer_type: str
        types of tokenizers that can be trained from scratch
        currently supported WordPiece, BPE, Unigram, WordLevel
    vocab_size: int
        number of tokens in a trained tokenizer
    hf_cache_dir: str
        path to a folder where hf tokenizers are stored
    from_old_flag : bool
        if True, use pre-trained tokenizer as a template.
    checkpoint_path : str
        path to pretrained tokenizer model.
    tokenizer_fname: str
        name for the file or folder where the trained
        tokenizer will be stored
    corpus_gen_batch_size: int
        size of batch for tokenizer training data corpus
        by deafult it is 1000
    """

    def __init__(
        self,
        fields: Union[str, List[str]],
        tokenizer_type: str = "",
        vocab_size: int = 1000,
        hf_cache_dir: str = "",
        from_old_flag: bool = False,
        checkpoint_path: str = None,
        tokenizer_fname: str = "tokenizer",
        corpus_gen_batch_size: int = 1000,
        special_tokens: List[str] = None,
    ):
        """Constructor method"""
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = fields

        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.from_old_flag = from_old_flag
        self.checkpoint_path = checkpoint_path
        self.tokenizer_fname = tokenizer_fname
        self.corpus_gen_batch_size = corpus_gen_batch_size
        self.special_tokens = special_tokens

        if hf_cache_dir:
            self.checkpoint_path = f"{hf_cache_dir}/{checkpoint_path}"

        self._data_write_config: DataWriteConfig = {
            "data_format": "parquet",
            "data_format_args": {"compression": "snappy", "use_dictionary": False},
        }
        self._artifacts_write_config: TokenizerTrainerArtifactsWriteConfig = {
            "vocab_format": "json",
            "vocab_format_args": {},
        }

    def set_write_config(
        self,
        data_config: DataWriteConfig = None,
        artifacts_config: TokenizerTrainerArtifactsWriteConfig = None,
    ):
        """Overwrite the default file writing configurations"""

        if data_config:
            self._data_write_config = data_config
        if artifacts_config:
            self._artifacts_write_config = artifacts_config

    @abstractmethod
    def run(self):
        """Abstract method"""
        pass


class CPUTokenizerTrainer(TokenizerTrainer):
    """TransformerTokenizer specific for CPU computation.

    Note
    ----
    This implementation of the TokenizerTrainer is specific for CPU computation.

    Attributes
    ----------

    field : str
        the name of the column containing text
    tokenizer_type: str
        types of tokenizers that can be trained from scratch
        currently supported WordPiece, BPE, Unigram, WordLevel
    vocab_size: int
        number of tokens in a trained tokenizer
    hf_cache_dir: str
        path to a folder where hf tokenizers are stored
    from_old_flag : bool
        if True, use pre-trained tokenizer as a template.
    checkpoint_path : str
        path to pretrained tokenizer model.
    tokenizer_fname: str
        name for the file or folder where the trained
        tokenizer will be stored
    corpus_gen_batch_size: int
        size of batch for tokenizer training data corpus
        by deafult it is 1000
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer_model = None

    def run(self, data: pa.Table, artifacts: dict) -> Tuple[pa.Table, dict]:
        """Runs tokenizer trainer based on the configuration used
        to create the object of the CPUTransformerTokenizer class

        Parameters
        ----------

        data : pyarrow.Table
            a pyarrow Table containing at least one list column containing
            text
        artifacts : dict
            artifacts are not consumed in this run method, but must be
            received to operate correctly in the pipeline run method

        Returns
        -------

        Tuple[pyarrow.Table, dict]
            The first position is a pyarrow.Table of pre-tokenized data.
            The second position is a dictionary of artifacts. The dict will
            contain keys for "embedding_matrix" and "id_to_token".
        """
        # Perform validations
        validate_pyarrow_table(data=data)
        validate_str_cols(fields=self.fields, data=data)

        # Use the Polars library to convert a text column into a list of lists
        # [each list here is a set of tokens]. The list of lists is called
        # df_words_lists. If there are multiple text columns, the df_words_list
        # is extended with contents of rows of the additional columns
        text_data = pl.from_arrow(data.select(self.fields))
        df_words_lists = []
        for field in self.fields:
            df_words_lists.extend(text_data[field].drop_nulls().to_list())

        # generator to iterate over all of the data
        # corpus geerator batch size
        def get_training_corpus():
            return (
                df_words_lists[i: i + self.corpus_gen_batch_size]
                for i in range(0, len(df_words_lists), self.corpus_gen_batch_size)
            )

        training_corpus = get_training_corpus()

        if self.from_old_flag:
            # load an existing tokenizer train a new one from the old one
            old_tokenizer = tokenizers_lib.load_hf_tokenizer(self.checkpoint_path)
            self.tokenizer_model = old_tokenizer.train_new_from_iterator(
                training_corpus, vocab_size=self.vocab_size
            )

        else:
            # train custom tokenizer from scratch, create a TrainableTokenizer
            # object with default values for special tokens and so on
            self.tokenizer_model = TrainableTokenizer(
                self.tokenizer_type, self.vocab_size, self.special_tokens
            )

            self.tokenizer_model.tokenizer.train_from_iterator(
                training_corpus, trainer=self.tokenizer_model.trainer
            )
            self.tokenizer_model = self.tokenizer_model.tokenizer

        # Set up the artifacts dict to return
        produced_artifacts = {
            "tokenizer_model": self.tokenizer_model,
            "vocab_size": self.vocab_size,
        }

        if self.from_old_flag:
            produced_artifacts["from_old_flag"] = self.checkpoint_path
        else:
            produced_artifacts["tokenizer_type"] = self.tokenizer_type

        return (data, produced_artifacts)

    def write_artifacts(self, write_path: str, artifacts: Union[dict, None]) -> None:
        """Write the oartifactsproduced by the embedding_generator

        Parameters
        ----------

        write_path : str
            Path is a directory where files will be written
        artifacts : Union[dict, None]
            Artifacts is a dictionary of artifacts produced in this step.
            Expected keys are: "id_to_token" and "embedding_matrix"
        """
        tokenizer_model = artifacts["tokenizer_model"]
        # Add the filename with the appropriate filetype to the write path

        if isinstance(tokenizer_model, Tokenizer):
            tokenizer_model_path = os.path.join(
                write_path,
                (f'{self.tokenizer_fname}.{self._artifacts_write_config["vocab_format"]}'),
            )
            tokenizer_model.save(tokenizer_model_path)

        else:
            tokenizer_model_path = os.path.join(write_path, f"{self.tokenizer_fname}")
            tokenizer_model.save_pretrained(tokenizer_model_path)

    def get_parameters(self) -> dict:
        """Retrive the embedding generetor object configuration

        Returns
        -------

        dict
            a dictionary representation of the embedding generetor object's attributes
        """

        params = vars(self).copy()
        params.pop("tokenizer_model")

        return params
