"""Clean text with custom sets of regular expressions"""

from abc import abstractmethod
from typing import List, Tuple, Union, Optional

import polars as pl
import pyarrow as pa

from bardi.nlp_engineering.regex_library.regex_lib import RegexSubPair
from bardi.nlp_engineering.utils.validations import (
    validate_pyarrow_table,
    validate_str_cols,
)
from bardi.pipeline import DataWriteConfig, Step


class Normalizer(Step):
    """Normalizer cleans and standardizes text input using regular expression
    substitutions. Lowercasing is also applied if desired.

    Note
    ----
    
    Avoid the direct instantiation of the Normalizer class and instead instantiate 
    one of the child classes depending on hardware configuration.

    Attributes
    ----------

    fields : Union[str, List[str]]
        The field or fields to be normalized.
    regex_set : List[RegexSubPair]
        List of regex substitutions to be applied.
    lowercase : Optional[bool]
        If True, lowercasing will be applied during normalization,
        defaults to True.
    """

    def __init__(
        self,
        fields: Union[str, List[str]],
        regex_set: List[RegexSubPair],
        lowercase: bool = True,
    ):
        """Constructor method
        """
        # Normalizer Configuration
        self.fields = fields
        self._data_write_config: DataWriteConfig = {
            "data_format": "parquet",
            "data_format_args": {"compression": "snappy", "use_dictionary": False},
        }  # Default write config
        self.lowercase = lowercase
        # List of regex substitutions to be applied
        self.regex_set: List[RegexSubPair] = regex_set

    @abstractmethod
    def run(self):
        """Abstract method
        """
        pass


class CPUNormalizer(Normalizer):
    """Normalizer class for cleaning and standardizing text input using regular expression substitutions.

    Note
    ----
    This implementation of the Normalizer is specific for CPU computation.

    Attributes
    ----------
    
    fields : Union[str, List[str]]
        The name of the column(s) containing text to be normalized.
    regex_set : List[RegexSubPair]
        A list of dictionaries with keys, 'regex_str' and 'sub_str', used to
        perform regular expression substitutions of the text.
    lowercase : Optional[bool] 
        If True, lowercasing will be applied during normalization.
        Default is True.
    """

    def __init__(self, *args, **kwargs):
        """Constructor method
        """
        super().__init__(*args, **kwargs)
        # Rust uses a different syntax for regex match groups
        # than standard (uses '$' instead of backslash)
        for regex_sub_pair in self.regex_set:
            regex_sub_pair["sub_str"] = regex_sub_pair["sub_str"].replace("\\", "$")

    def run(self, data: pa.Table, artifacts: Optional[dict] = None) -> Tuple[pa.Table, dict]:
        """Run the CPU-based normalizer method based on the configuration used to create
        the object of the CPUNormalizer class.

        Parameters
        ----------

        data : pyarrow.Table
            A pyarrow Table containing at least one text column of type string
            or large_string.
        artifacts : Optional[dict]
            Artifacts are not used in this run method but must be received
            to operate correctly in the pipeline run method.

        Returns
        -------

        Tuple[pyarrow.Table, dict]
            A tuple containing the pyarrow Table of cleaned data and an empty
            dictionary.
        """
        # Perform validations
        validate_pyarrow_table(data=data)
        validate_str_cols(fields=self.fields, data=data)

        def implement_regex_substitutions(df: pl.DataFrame) -> pl.DataFrame:
            """Reusable function to apply each regex substitution
            normalization to each string field
            
            Parameters
            ----------

            df : pl.DataFrame
                a DataFrame containing the columns specified in the Normalizer's
                `fields` attribute
            
            Returns
            -------
            pl.DataFrame
                a DataFrame with the specified fields having its text transformed using
                the regular expression substitution pairs in the Normalizer object's 'regex_set'
                attribute
            """

            for regex_sub_pair in self.regex_set:
                df = df.with_columns(
                    [
                        pl.col(field).str.replace_all(
                            pattern=regex_sub_pair["regex_str"],
                            value=regex_sub_pair["sub_str"],
                        )
                        for field in self.fields
                    ]
                )
            return df

        # Use the Polars library to apply the normalization methods
        # to each field of the Table that is specified in self.fields
        df = (
            pl.from_arrow(data)
            .with_columns(
                [
                    pl.col(field).str.to_lowercase()
                    for field in self.fields
                    if self.lowercase
                ]
            )
            .pipe(implement_regex_substitutions)
        )

        data = df.to_arrow()

        return (data, None)
