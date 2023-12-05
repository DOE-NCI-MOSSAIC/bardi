"""Split text columns into lists of tokens"""

from abc import abstractmethod
from typing import List, Tuple, Union

import polars as pl
import pyarrow as pa

from bardi.nlp_engineering.utils.validations import (validate_pyarrow_table,
                                                     validate_str_cols)
from bardi.pipeline import Step


class PreTokenizer(Step):
    """The pre tokenizer breaks down text into smaller units
    before futher tokenization is applied.

    Avoid the direct instantiation of the PreTokenizer class
    and instead instantiate one of the child classes depending
    on hardware configuration.

    Attributes:
        fields : Union[str, List[str]],
            the name of the column containing text
        split_pattern : str
            a specific pattern of characters used to divide a string
            into smaller segments or tokens.
            By default, the split is done on a single space character.

    Methods:
        run
        get_parameters
        set_write_config
        write_outputs
    """
    def __init__(self,
                 fields: Union[str, List[str]],
                 split_pattern: str = " "):
        """instantiate a pre tokenizer object"""
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = fields
        self.split_pattern = split_pattern

    @abstractmethod
    def run(self):
        """to be implemented in child class"""
        pass


class CPUPreTokenizer(PreTokenizer):
    """Implementation of the PreTokenizer for CPU computation

    Inherited Attributes:
        fields : str or List[str]
        split_pattern : str

    Attributes:
        None

    Methods:
        run
        get_parameters
        set_write_config
        write_outputs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, data: pa.Table,
            artifacts: dict = None) -> Tuple[pa.Table, Union[dict, None]]:
        """Runs a CPU-based pre-tokenizer method based on the configuration
        used to create the object of the CPUPreTokenizer class

        Arguments:
            data : pyarrow.Table
                a pyarrow Table containing at least one text column of type
                string or large_string
            artifacts : dict
                artifacts are not used in this run method, but must be received
                to operate correctly in the pipeline run method

        Returns:
            Tuple[pyarrow.Table, dict]
                The first position is a pyarrow.Table of pre-tokenized data
                The second position is a dictionary of artifacts. No artifacts
                are produced in this run method, so the second position will
                return None.
        """
        # Perform validations
        validate_pyarrow_table(data=data)
        validate_str_cols(fields=self.fields, data=data)

        # Split text fields into lists of tokens
        df = (pl.from_arrow(data)
              .with_columns(
                  [(pl.col(field)
                      .str.split(by=self.split_pattern)
                      .list.eval(pl.element().filter(pl.element() != ''))
                      .alias(field))
                   for field in self.fields]
                  )
              )

        data = df.to_arrow()

        return (data, None)
