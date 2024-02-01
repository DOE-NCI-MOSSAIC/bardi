"""Train a Word2Vec model and create a vocab and word embeddings"""

import os
from abc import abstractmethod
from multiprocessing import cpu_count
from os.path import isfile
from typing import List, TypedDict, Union, Tuple, Optional

import numpy as np
import polars as pl
import pyarrow as pa
from gensim.utils import RULE_DEFAULT, RULE_DISCARD
from gensim.models import Word2Vec

from bardi.data import data_handlers
from bardi.nlp_engineering.utils.validations import (
    validate_pyarrow_table,
    validate_list_str_cols,
)
from bardi.pipeline import DataWriteConfig, Step


class EmbeddingGeneratorArtifactsWriteConfig(TypedDict):
    """Indicates the keys and data types expected in an
    artifacts write config dict for the embedding generator
    if overwriting the default configuration."""

    vocab_format: str
    vocab_format_args: dict
    embedding_matrix_format: str
    embedding_matrix_format_args: dict


class EmbeddingGenerator(Step):
    """The embedding generator provides an interface to create
    word embeddings or vector representations of words (tokens).

    The embedding generator uses the Word2Vec model from the Gensim library.

    Note
    ----

    Avoid the direct instantiation of the PreTokenizer class
    and instead instantiate one of the child classes depending
    on hardware configuration.

    Attributes
    ----------

    fields : Union[str, List[str]]
        The name of the column(s) containing text to generate embeddings.
    load_saved_model : bool
        Whether to load a saved Word2Vec model or train a new one.
    checkpoint_path : str
        Path to the saved model checkpoint if `load_saved_model` is True.
    cores : int
        Number of CPU cores to use for training.
    min_word_count : int
        Ignore all words with a total frequency lower than this.
    window : int
        Maximum distance between the current and predicted word within a sentence.
    vector_size : int
        Dimensionality of the word vectors.
    sample : float
        The threshold for configuring which higher-frequency words are randomly downsampled.
    min_alpha : float
        Learning rate will linearly drop to `min_alpha` as training progresses.
    negative : int
        If > 0, specifies how many "noise words" should be drawn.
    epochs : int
        Number of iterations (epochs) over the corpus.
    seed : int
        Seed for the random number generator.
    vocab_exclude_list : List[str]
        List of words to force exclude from the vocabulary.
    """

    def __init__(
        self,
        fields: Union[str, List[str]],
        load_saved_model: bool = False,
        checkpoint_path: Optional[str] = None,
        cores: int = cpu_count(),
        min_word_count: int = 10,
        window: int = 5,
        vector_size: int = 300,
        sample: float = 6e-5,
        min_alpha: float = 0.007,
        negative: int = 20,
        epochs: int = 30,
        seed: int = 42,
        vocab_exclude_list: List[str] = [],
    ):
        """Constructor method
        """
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = fields
        self.load_saved_model = load_saved_model
        self.checkpoint_path = checkpoint_path
        self.cores = cores
        self.min_word_count = min_word_count
        self.window = window
        self.vector_size = vector_size
        self.sample = sample
        self.min_alpha = min_alpha
        self.negative = negative
        self.epochs = epochs
        self.seed = seed
        self.vocab_exclude_list = vocab_exclude_list
        self._data_write_config: DataWriteConfig = {
            "data_format": "parquet",
            "data_format_args": {"compression": "snappy", "use_dictionary": False},
        }
        self._artifacts_write_config: EmbeddingGeneratorArtifactsWriteConfig = {
            "vocab_format": "json",
            "vocab_format_args": {},
            "embedding_matrix_format": "npy",
            "embedding_matrix_format_args": {},
        }

    def set_write_config(
        self,
        data_config: Optional[DataWriteConfig] = None,
        artifacts_config: Optional[EmbeddingGeneratorArtifactsWriteConfig] = None,
    ):
        """Overwrite the default file writing configurations"""

        if data_config:
            self._data_write_config = data_config
        if artifacts_config:
            self._artifacts_write_config = artifacts_config

    @abstractmethod
    def run(self):
        """Abstract method
        """
        pass


class CPUEmbeddingGenerator(EmbeddingGenerator):
    """The embedding generator provides an interface to create
    word embeddings or vector representations of words (tokens).

    The embedding generator uses the Word2Vec model from the Gensim library.

    Note
    ----
    This implementation of the EmbeddingGenerator is specific for CPU computation.

    Attributes
    ----------

        fields : Union[str, List[str]]
            The name of the column(s) containing text to be considered
            in the vocab and used in Word2Vec.
        load_saved_model : bool
            If True, use a pre-trained Word2Vec model.
        checkpoint_path : str
            Path to the Word2Vec model checkpoint.
        cores : int
            Number of cores to run the Word2Vec model on.
        min_word_count : int
            Ignores all words with total frequency lower than this.
        window : int
            Maximum distance between the current and predicted word.
        vector_size : int
            Output embedding size.
        sample : float
            The threshold for configuring which high-frequency
            words are randomly downsampled, use range (0, 1e-5).
        min_alpha : float
            Learning rate will linearly drop to min_alpha
            as training progresses.
        negative : int
            If > 0, negative sampling will be used.
        epochs : int
            Total number of iterations of all training data in
            the training of the Word2Vec model.
        seed : int
            Seed for the random number generator. For deterministic
            run, you need thread = 1 (aka CPU core)
            and PYTHONHASHSEED.
        vocab_exclude_list : List[str]
            Provide a list of tokens that may be present in the
            text that you would like to exclude from the vocab and
            from Word2Vec.
    """

    def __init__(self, *args, **kwargs):
        """Constructor method
        """
        super().__init__(*args, **kwargs)

        self.w2v_model = None
        self.embedding_matrix = None
        self.id_to_token = None

    def run(self, data: pa.Table, artifacts: dict) -> Tuple[pa.Table, dict]:
        """Runs a CPU-based embedding generator run method based on the
        configuration used to create the object of the CPUEmbeddingGenerator
        class

        Parameters
        ----------

        data : pyarrow.Table
            A pyarrow Table containing at least one list column containing text.
        artifacts : dict
            Artifacts are not consumed in this run method, but must be
            received in the method to operate correctly in the pipeline run method.

        Returns
        -------

        Tuple[pyarrow.Table, dict]
            The first position is a pyarrow.Table of pre-tokenized data.
            The second position is a dictionary of artifacts. The dict will
            contain keys for "embedding_matrix" and "id_to_token".
        """
        # Perform validations
        validate_pyarrow_table(data=data)
        validate_list_str_cols(fields=self.fields, data=data)

        # Use the Polars library to convert a text column into a list of lists
        # [each list here is a set of tokens]. The list of lists is called
        # df_words_lists. If there are multiple text columns, the df_words_list
        # is extended with contents of rows of the additional columns.
        text_data = pl.from_arrow(data.select(self.fields))
        df_words_lists = []
        for field in self.fields:
            df_words_lists.extend(text_data[field].drop_nulls().to_list())

        if self.load_saved_model and isfile(self.checkpoint_path):
            self.w2v_model = Word2Vec.load(self.checkpoint_path)
        else:
            # Configure Word2Vec
            self.w2v_model = Word2Vec(
                min_count=self.min_word_count,
                window=self.window,
                vector_size=self.vector_size,
                sample=self.sample,
                min_alpha=self.min_alpha,
                negative=self.negative,
                workers=self.cores,
                seed=self.seed,
            )

            # Word2Vec vocabulary trim rule
            def vocab_trim_rule(word: str, count: int, min_count: int):
                if count < min_count:
                    return RULE_DISCARD
                elif word in self.vocab_exclude_list:
                    return RULE_DISCARD
                else:
                    return RULE_DEFAULT

            # Train Word2Vec Model
            self.w2v_model.build_vocab(df_words_lists, trim_rule=vocab_trim_rule)
            self.w2v_model.train(
                df_words_lists,
                total_examples=self.w2v_model.corpus_count,
                epochs=self.epochs,
            )
        # If a checkpoint path was set, the word2vec model will be saved
        if self.checkpoint_path:
            self.w2v_model.save(self.checkpoint_path)

        # Sort the vocabulary
        sorted_tokens = sorted(self.w2v_model.wv.index_to_key)
        sorted_embeddings = []

        for token in sorted_tokens:
            sorted_embeddings.append(
                self.w2v_model.wv.vectors[self.w2v_model.wv.key_to_index[token]]
            )

        # Mapping between an index in the embedding matrix and a token
        self.id_to_token = {
            token_id: token
            for token_id, token in enumerate(["<pad>"] + sorted_tokens + ["<unk>"])
        }

        # Create embedding matrix with embeddings for <pad> and <unk> tokens
        pad_token_emb = np.zeros((1, self.vector_size))
        unkn_token_emb = np.random.random((1, self.vector_size)) * 0.01
        self.embedding_matrix = np.append(
            np.append(pad_token_emb, sorted_embeddings, axis=0), unkn_token_emb, axis=0
        )

        # Set up the artifacts dict to return
        produced_artifacts = {
            "embedding_matrix": self.embedding_matrix,
            "id_to_token": self.id_to_token,
        }

        return (data, produced_artifacts)

    def write_artifacts(self, write_path: str, artifacts: dict) -> None:
        """Write the artifacts produced by the embedding generator.

        Parameters
        ----------
        write_path : str
            Path is a directory where files will be written.
        artifacts : dict
            Artifacts is a dictionary of artifacts produced in this step.
            Expected keys are: "id_to_token" and "embedding_matrix".
        """
        id_to_token = artifacts["id_to_token"]
        # Add the filename with the appropriate filetype to the write path
        id_to_token_path = os.path.join(
            write_path,
            (f"id_to_token" f'.{self._artifacts_write_config["vocab_format"]}'),
        )
        # Call the data handler write_file function
        data_handlers.write_file(
            data=id_to_token,
            path=id_to_token_path,
            format=self._artifacts_write_config["vocab_format"],
            **self._artifacts_write_config["vocab_format_args"],
        )

        embedding_matrix = artifacts["embedding_matrix"]
        # Add the filename with the appropriate filetype to the write path
        embedding_matrix_path = os.path.join(
            write_path,
            (
                f"embedding_matrix"
                f'.{self._artifacts_write_config["embedding_matrix_format"]}'
            ),
        )
        # Call the data handler write_file function
        data_handlers.write_file(
            data=embedding_matrix,
            path=embedding_matrix_path,
            format=self._artifacts_write_config["embedding_matrix_format"],
            **self._artifacts_write_config["embedding_matrix_format_args"],
        )

    def get_parameters(self) -> dict:
        """Retrieve the embedding generator object configuration.

        Returns
        -------
        dict
            A dictionary representation of the EmbeddingGenerator object's attributes.
        """

        params = vars(self).copy()
        params["vocab_size"] = len(params["id_to_token"].keys())
        params.pop("id_to_token")
        params.pop("embedding_matrix")
        params["w2v_model"] = str(type(self.w2v_model))

        return params
