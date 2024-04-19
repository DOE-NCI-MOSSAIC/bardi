"""Apply a tokenizer to the provided text fields"""

from abc import abstractmethod
from multiprocessing import cpu_count
from typing import List, Optional, Union

import datasets
import polars as pl
import pyarrow as pa

from bardi.nlp_engineering.utils import tokenizers_lib
from bardi.nlp_engineering.utils.polars_utils import retain_inputs
from bardi.nlp_engineering.utils.validations import validate_pyarrow_table, validate_str_cols
from bardi.pipeline import DataWriteConfig, Step


class TokenizerEncoder(Step):
    """The tokenizer encoder uses a trained tokenizer
    to split text into tokens.

    Note
    ----

    Avoid the direct instantiation of the TokenizerEncoder class
    and instead instantiate one of the child classes depending
    on hardware configuration.

    Attributes
    ----------

    fields : Union[str, List[str]],
        the name of the column containing a list of tokens
        that will be mapped to integers using a vocab
    return_tensors: str
        type of tensors to return, 'np' for Numpy arrays,
        'pt' for PyTorch tensors or 'tf' for TensorFlow
    concat_fields : bool
        whether the text fields should be concatenate into a single
        text field, defaults to False
    retain_input_fields : Optional[bool]
        If True, will retain the original contents of the fields specified in
        `fields` under the new names of: `TokenizerEncoder__<field>`
    retain_concat_field : Optional[bool]
        If True, will retain the concatenation of the fields specified in
        `fields` under the name specified in `field_rename` or 'text' if not
        specified. If `concat_fields` is not True, this parameter will
        have no effect
    field_rename : Optional[str]
        optional ability to rename the supplied field with
        the field_rename value
    hf_cache_dir : Optional[str]
        local directory where the HF pretrained tokenizers are stored
    model_name : Optional[str]
        name of a tokenizer file or folder. Not required if TokenizerTrainer
        is a prior step - tokenizer will be passed in artifacts
    cores : Optional[int]
        number of CPU cores for multithreading the tokenizer
    tokenizer_params : Optional[dict]
        provide fine-grained customization for any valid HuggingFace Tokenizer parameter
        through a dictionary
    tokenizer_model : transformers.PreTrainedTokenizerBase
        Tokenizer object passed through artifacts from TokenizerTrainer or
        read from file specified in `model_name`
    """

    def __init__(
        self,
        fields: Union[str, List[str]],
        return_tensors: str = "np",
        concat_fields: bool = False,
        retain_input_fields: bool = False,
        retain_concat_field: bool = False,
        field_rename: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        cores: Optional[int] = None,
        tokenizer_params: Optional[dict] = None,
    ):
        """Constructor method"""
        self.fields = fields
        if isinstance(fields, str):
            self.fields = [fields]
        self.field_rename = "text"
        if field_rename:
            self.field_rename = field_rename
        self.retain_input_fields = retain_input_fields
        self.retain_concat_field = retain_concat_field
        self.model_name = model_name
        if hf_cache_dir:
            self.model_name = f"{hf_cache_dir}/{model_name}"
        self.concat_fields = concat_fields
        self.cores = cpu_count()
        if cores:
            self.cores = cores
        self.tokenizer_params = {}
        if tokenizer_params:
            self.tokenizer_params = tokenizer_params
        self.return_tensors = return_tensors
        self.tokenizer_model = None
        self._data_write_config: DataWriteConfig = {
            "data_format": "parquet",
            "data_format_args": {"compression": "snappy", "use_dictionary": False},
        }  # Default write config

    @abstractmethod
    def run(self):
        """Abstract method"""
        pass


class CPUTokenizerEncoder(TokenizerEncoder):
    """Implementation of the TokenizerEncoder for CPU computation

    Note
    ----
    This implementation of the TokenizerEncoder is specific for CPU computation.

    Attributes
    ----------

    fields : Union[str, List[str]],
        the name of the column containing a list of tokens
        that will be mapped to integers using a vocab
    return_tensors: str
        type of tensors to return, 'np' for Numpy arrays,
        'pt' for PyTorch tensors or 'tf' for TensorFlow
    concat_fields : bool
        whether the text fields should be concatenate into a single
        text field, defaults to False
    retain_input_fields : Optional[bool]
        If True, will retain the original contents of the fields specified in
        `fields` under the new names of: `CPUTokenizerEncoder_input__<field>`
    retain_concat_field : Optional[bool]
        If True, will retain the concatenation of the fields specified in
        `fields` under the name specified in `field_rename` or 'text' if not
        specified. If `concat_fields` is not True, this parameter will
        have no effect
    field_rename : Optional[str]
        optional ability to rename the supplied field with
        the field_rename value
    hf_cache_dir : Optional[str]
        local directory where the HF pretrained tokenizers are stored
    model_name : Optional[str]
        name of a tokenizer file or folder. Not required if TokenizerTrainer
        is a prior step - tokenizer will be passed in artifacts
    cores : Optional[int]
        number of CPU cores for multithreading the tokenizer
    tokenizer_params : Optional[TokenizerConfig]
        provide fine-grained settings for applying tokenizer
    tokenizer_model : transformers.PreTrainedTokenizerBase
        Tokenizer object passed through artifacts from TokenizerTrainer or
        read from file specified in `model_name`
    """

    def __init__(self, *args, **kwargs):
        """Constructor method"""
        super().__init__(*args, **kwargs)

    def run(self, data: pa.Table, artifacts: dict = None) -> pa.Table:
        """Run a tokenizer encoder based on provided configuration

        The tokenizer encoder relies on receiving a tokenizer to apply to
        the text. The tokenizer can be supplied in multiple ways:
          - referencing the model_name at object creation
          - contained in the pipeline artifacts dictionary passed
            to the run method referenced by the key, 'tokenizer_model'

        Parameters
        ----------

        data : PyArrow Table
            The data to be processed. The data must contain the
            column specified by field at object creation
        artifacts : dict
            A dictionary of pipeline artifacts which contains
            a tokenizer referenced by the key, 'tokenizer_model'

        Returns
        -------

        Tuple[PyArrow Table, dict]
            The first element holding a PyArrow Table of data
            processed with the tokenizer encoder. The second
            element of the tuple intended for artifacts is None.

        Raises
        ------

        AttributeError
            The tokenizer wasn't supplied either at object creation
            or to the run method
        TypeError
            The run method was not supplied a PyArrow Table
        """

        # Perform validations
        # columns are of type string or List[str] -> pretokenization
        # was not applied
        validate_pyarrow_table(data=data)
        validate_str_cols(fields=self.fields, data=data)

        # Check if a tokenizer model was passed through the artifacts dict
        if artifacts:
            if "tokenizer_model" in artifacts.keys():
                self.tokenizer_model = artifacts["tokenizer_model"]

        if self.model_name:
            self.tokenizer_model = tokenizers_lib.load_tokenizer(self.model_name)

        if not self.tokenizer_model:
            raise AttributeError(
                "A tokenizer needs to be provided to this step. Either provide "
                "the model name in the `model_name` parameter at TokenizerEncoder "
                "instantiation or train a tokenizer in this pipeline before this "
                "step with the bardi TokenizerTrainer class."
            )

        # Set tokenizers params
        self.tokenizer_model = tokenizers_lib.set_tokenizer_params(
            self.tokenizer_model, **self.tokenizer_params
        )

        # Retain input columns in original form if desired
        df = pl.from_arrow(data)

        # Concat the fields if desired
        if self.concat_fields and len(self.fields) > 1:
            # columns are of type string, some of them are null
            # concat their values into a single string
            df = (
                df.pipe(
                    retain_inputs, self.retain_input_fields, self.fields, self.__class__.__name__
                )
                .with_columns(
                    pl.concat_str(
                        [pl.col(field).fill_null("") for field in self.fields],
                        separator=" ",
                    ).alias("text")
                )
                .with_columns(
                    [pl.col("text").str.replace_all(pattern=r"\s\s+", value=" ").str.strip_chars()]
                )
                .drop([field for field in self.fields])
            )

            # concat field is called 'X', rename if
            # other name provided, defaults to 'X'
            df = df.rename({"text": self.field_rename})

        # if concat was not run and a single field was provided
        # the field can be renamed
        elif len(self.fields) == 1:
            df = df.rename({self.fields[0]: self.field_rename})

        # Apply tokenizer - we could support more params
        def tokenize(text):
            # add check for len here or filter dataframe first
            return self.tokenizer_model(
                text,
                padding=True,
                truncation=True,
                max_length=self.tokenizer_model.model_max_length,
                return_tensors=self.return_tensors,
            )

        data = df.to_arrow()
        hf_dataset = datasets.Dataset(arrow_table=data)

        # there is a single column to process
        if self.concat_fields or len(self.fields) == 1:
            hf_dataset = hf_dataset.map(
                lambda row: tokenize(row[self.field_rename]),
                batched=True,
                num_proc=self.cores,
            )
            if not self.retain_concat_field:
                hf_dataset = hf_dataset.remove_columns(self.field_rename)

        # there are multiple columns each tokenized separately
        # with the same tokenizer
        else:
            for field in self.fields:
                hf_dataset = (
                    hf_dataset.map(
                        lambda row: tokenize(row[field]), batched=True, num_proc=self.cores
                    )
                    .rename_column("input_ids", f"{field}_input_ids")
                    .rename_column("attention_mask", f"{field}_attention_mask")
                    .remove_columns(field)
                )
        data = hf_dataset.data.table

        return (data, None)

    def get_parameters(self):
        """Retrive the post-processor object configuration
        Does not return the mapping (vocab) as it can be large

        Returns
        -------

        dict
            a dictionary representation of the post-processor's attributes
        """
        params = vars(self).copy()
        params["tokenizer_model"] = str(params["tokenizer_model"])

        return params
