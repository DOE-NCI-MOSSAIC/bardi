"""Utilities for the tokenizers' support"""
from os.path import isdir
from typing import Union

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase


class TrainableTokenizer:
    """
    A class to reporesent a trainable tokenizer.
    It currently supports training the following
    HF tokenizers from scratch
    of type:
        - WordPiece
            subword tokenizer algorithm, uses greedy algorithm that tries
            to build long words first, splitting in multiple tokens when
            entire words do not exist in the vocabulary. Uses ## prefix
            to identify tokens that are part of a word (not starting a word)
        - BPE
            subword tokenization algorithm. BPE starts with characters
            then merges those that are seen next to each other most often,
            thus creating new tokens from most frequent pairs.
        - Unigram
            subword tokenization algorithm, works by trying to identify
            the best set of subword tokens to maximize the probability
            of a given sentence.
        - WordLevel
            simply maps words to IDs, required large vocabulary size
    """

    def __init__(self, tokenizer_type, voc_size, special_tokens):
        self.tokenizer_type = tokenizer_type
        self.voc_size = voc_size
        self.special_tokens = special_tokens
        self.tokenizer = None
        self.trainer = None
        self.subword_tokenizers = ["WordPiece", "BPE", "Unigram", "WordLevel"]

        if self.tokenizer_type == "WordPiece":
            # default special tokens for WordPiece tokenizer
            if self.special_tokens is None:
                self.special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
            self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

            self.trainer = trainers.WordPieceTrainer(
                vocab_size=self.voc_size,
                special_tokens=self.special_tokens,
            )

        elif self.tokenizer_type == "BPE":
            self.special_tokens = ["<|endoftext|>"]

            self.tokenizer = Tokenizer(models.BPE())
            self.trainer = trainers.BpeTrainer(
                vocab_size=self.voc_size,
                special_tokens=self.special_tokens,
            )

        elif self.tokenizer_type == "Unigram":
            self.special_tokens = ["<|endoftext|>"]

            self.tokenizer = Tokenizer(models.Unigram())
            self.trainer = trainers.UnigramTrainer(
                vocab_size=self.voc_size,
                special_tokens=self.special_tokens,
            )

        elif self.tokenizer_type == "WordLevel":
            self.special_tokens = ["<|endoftext|>"]

            self.tokenizer = Tokenizer(models.WordLevel())
            self.trainer = trainers.WordLevelTrainer(
                vocab_size=self.voc_size,
                special_tokens=self.special_tokens,
            )
        # subword tokenizers require a pretokenizer
        # for now all of them will use custom Whitespace
        if self.tokenizer_type in self.subword_tokenizers:
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        return


def load_hf_tokenizer(path: str) -> AutoTokenizer:
    """
    Loads HF tokenizer for 'train a new tokenizer from an old' one option.

    Parameters
    ----------

    path : str
        path to a directory with HF tokenizer's config
        json files. We expect:
            - tokenizer.json
            - tokenizer_config.json
            - special_tokens_map.json

    Returns
    -------

    tokenizer : AutoTokenizer
        HF AutoTokenizer object
    """
    tokenizer = AutoTokenizer.from_pretrained(path)

    return tokenizer


def load_tokenizer(path: str) -> Union[Tokenizer, AutoTokenizer]:
    """
    Loads any HF tokenizer including model agnostic
    custom tokenizer. This will be extended to support
    SentencePiece tokenizers

    Parameters
    ----------

    path : str
        path to directory or to .json file
        with a tokenizer info

    Returns
    -------

    tokenizer : Union[Tokenizer, AutoTokenizer]
        return object of class Tokenizer
        or AutoTokenizer depending on the provided
        files
    """
    if isdir(path):
        return AutoTokenizer.from_pretrained(path)

    return Tokenizer.from_file(path)


def set_tokenizer_params(
    tokenizer: Union[AutoTokenizer, Tokenizer],
    model_max_length: int = 4096,
    unk_token: str = "[UNK]",
    pad_token: str = "[PAD]",
    *args, **kwargs
) -> AutoTokenizer:
    """Overwrites the defaults AutoTokenizer settings.
    If a user provides a TokenizerConfigs object then
    AutoTokenizer's settings are set to the values provided.
    If the user does not provide any values, the setting that
    come from tokenizer_config.json are used. If required values
    like model_max_length or pad_token are missing in the tokenizer_config.json
    then they are set to defaults.

    model_max_length = 4096
    unk_token = "[UNK]"
    pad_token = "[PAD]"

    Parameters
    ----------

    tokenizer : AutoTokenizer
        A HuggingFace tokenizer object
    model_max_length : int
        Defaults to 4096
    unk_token : str
        Defaults to
    pad_token : str
        Defaults to
    *args
        Variable length argument list
    **kwargs
        Arbitraty keyword argument list for any valid HuggingFace PreTrainedTokenizer
        parameters

    Returns
    -------

    tokenizer : AutoTokenizer
    """

    # Setting params for a custom trained tokenizer
    if isinstance(tokenizer, Tokenizer):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            max_length=model_max_length,
            unk_token=unk_token,
            pad_token=pad_token,
            *args, **kwargs
        )

    # Setting params for an HF PreTrainedTokenizer
    if isinstance(tokenizer, PreTrainedTokenizerBase):
        tokenizer.model_max_length = model_max_length

        # some pretrained tokenizers have model_max_size set
        # to  1000000000000000019884624838656 which is equivalent to not set
        if tokenizer.model_max_length > 1000000000000000:
            tokenizer.model_max_length = model_max_length

    # Add special tokens if they were not specified already
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": pad_token})
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({"unk_token": unk_token})

    return tokenizer
