"""Clean text with custom sets of regular expressions"""

from abc import abstractmethod
from typing import List, Tuple, Union

import polars as pl
import pyarrow as pa

from bardi.nlp_engineering.regex_library.regex_lib import RegexSubPair
from bardi.nlp_engineering.utils.validations import (validate_pyarrow_table,
                                                     validate_str_cols)
from bardi.pipeline import DataWriteConfig, Step


class Normalizer(Step):
    """Normalizer cleans and standardizes text input using regular expression
    substitutions. Lowercasing is also applied if desired.

    Avoid the direct instantiation of the PreTokenizer class
    and instead instantiate one of the child classes depending
    on hardware configuration.
    """
    def __init__(self,
                 fields: Union[str, List[str]],
                 regex_set: List[RegexSubPair],
                 lowercase: bool = True,
                 ):
        # Normalizer Configuration
        self.fields = fields
        self._data_write_config: DataWriteConfig = {
            "data_format": 'parquet',
            "data_format_args": {"compression": 'snappy',
                                 "use_dictionary": False}
        }  # Default write config
        self.lowercase = lowercase
        # List of regex substitutions to be applied
        self.regex_set: List[RegexSubPair] = regex_set

    @abstractmethod
    def run(self):
        """to be implemented in child class"""
        pass


class CPUNormalizer(Normalizer):
    """Normalizer cleans and standardizes text input using regular expression
    substitutions. Lowercasing is also applied if desired.

    Normalizer specific for CPU computation.

    Attributes:
        fields : Union[str, List[str]]
            the name of the column(s) containing text to be normalized
        regex_set : List[RegexSubPair]
            a list of dicionaries with keys, 'regex_str' and 'sub_str' used to
            perform regular expression substitutions of the text
        lowercase : bool
            whether or not the text will be lowercased

    Methods:
        run : run the step's primary function
        get_parameters : get a dictionary representation of the step object
        set_write_config : Alter the default file writing configuration
        write_outputs : Write output data to a file
    """
    def __init__(self, *args, **kwargs):
        """
        Keyword Arguments:
            fields : Union[str, List[str]]
                the name of the column(s) containing text to be normalized
            regex_set : List[RegexSubPair]
                a list of dicionaries with keys, 'regex_str' and 'sub_str' used to
                perform regular expression substitutions of the text
            lowercase : bool
                whether or not the text will be lowercased
        """
        super().__init__(*args, **kwargs)
        # Rust uses a different syntax for regex match groups
        # than standard (uses '$' instead of backslash)
        for regex_sub_pair in self.regex_set:
            regex_sub_pair['sub_str'] = regex_sub_pair['sub_str'].replace("\\", "$")

    def run(self, data: pa.Table,
            artifacts: dict = None) -> Tuple[pa.Table, dict]:
        """Runs a CPU-based normalizer method based on the configuration
        used to create the object of the CPUNormalizer class

        Arguments:
            data : pyarrow.Table
                a pyarrow Table containing at least one text column of type
                string or large_string
            artifacts : dict
                artifacts are not used in this run method, but must be received
                to operate correctly in the pipeline run method

        Returns:
            pyarrow.Table of cleaned data
        """
        # Perform validations
        validate_pyarrow_table(data=data)
        validate_str_cols(fields=self.fields, data=data)

        def implement_regex_substitutions(df: pl.DataFrame) -> pl.DataFrame:
            """Reusable function to apply each regex substitution
            normalization to each string field"""

            for regex_sub_pair in self.regex_set:
                df = df.with_columns(
                    [pl.col(field).str.replace_all(
                        pattern=regex_sub_pair['regex_str'],
                        value=regex_sub_pair['sub_str']) for field in self.fields])
            return df

        # Use the Polars library to apply the normalization methods
        # to each field of the Table that is specified in self.fields
        df = (pl.from_arrow(data)
              .with_columns([
                  pl.col(field).str.to_lowercase()
                  for field in self.fields if self.lowercase])
              .pipe(implement_regex_substitutions))

        data = df.to_arrow()

        return (data, None)
