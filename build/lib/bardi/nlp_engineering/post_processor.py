"""Apply a vocab mapping converting a list of tokens in a column
into a list of integers"""

from abc import abstractmethod
from math import ceil
from sys import getsizeof
from typing import List, Union

import numpy as np
import polars as pl
import pyarrow as pa

from bardi.data.utils.pyarrow_utils import chunk_pyarrow_table
from bardi.nlp_engineering.utils.validations import (
    validate_list_str_cols,
    validate_pyarrow_table,
)
from bardi.pipeline import DataWriteConfig, Step


class PostProcessor(Step):
    """The post processor maps a vocab to a list of tokens

    Note
    ----

    Avoid the direct instantiation of the PostProcessor class
    and instead instantiate one of the child classes depending
    on hardware configuration

    Attributes
    ----------

    fields : Union[str, List[str]]
        the name of the column containing a list of tokens
        that will be mapped to integers using a vocab
    field_rename : str
        optional ability to rename the supplied field with
        the field_rename value
    id_to_token : dict
        optional vocabulary in the form of {id: token} that
        will be used to map the tokens to integers. This
        is optional for the construction of the object, and
        can alternatively be provided in the run method.
        This flexibility handles the use of a pre-existing
        vocab versus creating a vocab during a pipeline run.
    concat_fields : bool
        indicate if you would like for fields to be concatenated
        into a single column or left as separate columns
    """

    def __init__(
        self,
        fields: Union[str, List[str]],
        field_rename: str = None,
        id_to_token: dict = None,
        concat_fields: bool = False,
    ):
        """Constructor method
        """
        self.fields = fields
        if isinstance(fields, str):
            self.fields = [fields]

        self.field_rename = "X"
        if field_rename:
            self.field_rename = field_rename

        self.mapping = None
        if id_to_token:
            # the actual application of the mapping requires
            # flipping the keys and values to have {token: id}
            self.mapping = {token: int(id) for id, token in id_to_token.items()}

        self.concat_fields = concat_fields
        self._data_write_config: DataWriteConfig = {
            "data_format": "parquet",
            "data_format_args": {"compression": "snappy", "use_dictionary": False},
        }  # Default write config

    @abstractmethod
    def run(self):
        """Abstract method
        """
        pass


class CPUPostProcessor(PostProcessor):
    """The post processor maps a vocab to a list of tokens

    Note
    ----

    This implementation of the PostProcessor is specific for CPU computation.

    Attributes
    ----------

    fields : Union[str, List[str]]
        the name of the column containing a list of tokens
        that will be mapped to integers using a vocab
    field_rename : str
        optional ability to rename the supplied field with
        the field_rename value
    id_to_token : dict
        optional vocabulary in the form of {id: token} that
        will be used to map the tokens to integers. This
        is optional for the construction of the object, and
        can alternatively be provided in the run method.
        This flexibility handles the use of a pre-existing
        vocab versus creating a vocab during a pipeline run.
    concat_fields : bool
        indicate if you would like for fields to be concatenated
        into a single column or left as separate columns
    """

    def __init__(self, *args, **kwargs):
        """Constructor method
        """
        super().__init__(*args, **kwargs)

    def run(
        self,
        data: pa.Table,
        artifacts: dict = None,
        id_to_token: dict = None,
    ) -> pa.Table:
        """Run a post processor using CPU computation

        The post processor relies on receiving a vocab to map. The
        vocab can be supplied in multiple ways:
          - id_to_token at object creation
          - contained in the pipeline artifacts dictionary passed
            to the run method referenced by the key, 'id_to_token'
          - id_to_token in the run method

        Attributes
        ----------

        data : PyArrow Table
            The data to be processed. The data must contain the
            column specified by field at object creation
        artifacts : dict
            A dictionary of pipeline artifacts which contains
            a vocab referenced by the key, 'id_to_token'
        id_to_token : dict
            If a vocab wasn't passed at object creation or
            through the pipeline artifacts dict, then it
            must be passed here as a final option

        Returns
        -------

        Tuple(PyArrow Table, dict)
            A tuple with:
                - the first element holding a PyArrow Table of data
                processed with the post processor
                - the second element of the tuple intended for
                artifacts is None

        Raises
        ------

        AttributeError
            The vocab (id_to_token) wasn't supplied
            either at object creation or to the run method
        TypeError
            The run method was not supplied a PyArrow Table
        """

        # Perform validations
        validate_pyarrow_table(data=data)
        validate_list_str_cols(fields=self.fields, data=data)

        # Check if vocab was passed through the artifacts dict
        if "id_to_token" in artifacts.keys():
            id_to_token = artifacts["id_to_token"]

        # Check if a vocab was passed directly to the method -
        # the route used if a vocab was created during script execution
        if id_to_token:
            self.mapping = {token: id for id, token in id_to_token.items()}
        # Check if a vocab was referenced in the initial object creation -
        # the route used if there was a pre-existing vocab
        elif not self.mapping:
            # If a vocab wasn't referenced in either place, then there
            # isn't a vocab to use in this run method
            raise AttributeError(
                "id_to_token must be provided to either "
                "the PostProcessor object instantiation "
                "or to the run method"
            )

        # Grab the <unk> id from the supplied vocab
        try:
            self.unk_id = self.mapping["<unk>"]
        except KeyError:
            self.unk_id = None  # If there isn't an unk id

        # Map tokens to ids using the supplied vocab.
        # If a token encountered isn't in the vocab, it defaults to
        # the <unk> value. The field passed is renamed to 'X'

        # This step is memory intensive, so the data is split into
        # 128 MB chunks and operated upon to avoid OoM errors
        data_size = getsizeof(data) * 1e-6
        chunks = ceil(data_size / 128)
        column_order = data.column_names
        tables = chunk_pyarrow_table(
            data=data, row_count=data.num_rows, min_batches=chunks
        )

        self.mapping[None] = None

        for i, table in enumerate(tables):
            chunk_df = pl.from_arrow(table)
            # create a unique id for each row in the chunk
            # this is needed for exploding and grouping
            # the data for efficinet execution
            index_list = np.arange(0, chunk_df.height)
            index_col = pl.Series(index_list)
            chunk_df = chunk_df.with_columns(index_col.alias("unique_id"))

            for field in self.fields:
                meta_df = chunk_df.drop(field)
                tmp_df = (
                    chunk_df.select(["unique_id", field])
                    .explode(field)
                    .with_columns(
                        pl.col(field).map_dict(
                            self.mapping, default=self.unk_id, return_dtype=pl.Int64()
                        )
                    )
                    .group_by("unique_id", maintain_order=True)
                    .agg(pl.col(field))
                )

                chunk_df = meta_df.join(tmp_df, on="unique_id", how="left")
            if i == 0:
                df = chunk_df.select(*column_order)
            else:
                df = df.vstack(chunk_df.select(*column_order))

        # concat the fields if desired
        if self.concat_fields and len(self.fields) > 1:
            # Concatenate the values in the fields.
            # Fields at this point are assumed to be large lists

            current_field = self.fields[0]
            for i in range(len(self.fields) - 1):
                next_field = self.fields[i + 1]
                df = df.with_columns(
                    pl.col(current_field)
                    .list.concat(next_field)
                    .list.drop_nulls()
                    .alias("X")
                ).drop(next_field)
                current_field = "X"
            df = df.drop(self.fields[0])

            # concat field is called 'X', rename if
            # other name provided, defaults to 'X'
            df = df.rename({"X": self.field_rename})

        # if concat was not run and a single field was provided
        # the field can be renamed
        elif len(self.fields) == 1:
            df = df.rename({self.fields[0]: self.field_rename})

        data = df.to_arrow()

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
        params.pop("mapping")
        return params
