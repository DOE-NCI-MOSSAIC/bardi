"""Dataset class definition and data handler functions for loading datasets from various sources"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Union

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.dataset as ds
import pyarrow.feather as feather
import pyarrow.parquet as pq
from pyarrow import orc

from bardi.data.utils.pyarrow_utils import chunk_pyarrow_table


class Dataset:
    """A dataset object handles data in the form of columns and rows

    Under the hood it uses a PyArrow Table as it is a modern and efficient
    starting point for both CPU & GPU workflows.

    Attributes
    ----------

    data : PyArrow.Table | List[PyArrow.Table]
        The data table
    origin_query : str
        If a SQL data source was used by a data_handler function, the SQL query is
        recorded here for reproducibility and data provenance.
    origin_file_path : str
        If a file was used as the data source by a data_handler function, the filepath
        is recorded for reproducibility and data provenance.
    origin_format : str
        The format of the data source
    origin_row_count : int
        The total row count of the original dataset
    """

    def __init__(self):
        self.date = str(datetime.now())
        self.data = None
        self.origin_query = None
        self.origin_file_path = None
        self.origin_format = None
        self.origin_row_count = None

    def get_parameters(self) -> dict:
        params = vars(self).copy()
        params["data"] = self.data.column_names
        params["origin_query"] = (
            " ".join(self.origin_query.split()) if self.origin_query else "None"
        )
        return params


# ======= Loaders ========


def from_file(
    source: Union[str, List[str]], format: str, min_batches: int = None, *args, **kwargs
) -> Dataset:
    """Create a bardi Dataset object from a file source

    Accepted file types are: parquet, ipc, arrow, feather, csv, and orc

    The function utilizes PyArrow's dataset API to read files, and thus all
    keyword arguments available for its API are available here.

    Parameters
    ----------

    source : str, List[str]
        Path to a single file, or list of paths
    format : str
        Currently ["parquet", "ipc", "arrow", "feather", "csv", "orc"] are
        supported
    min_batches : int
        An integer number used to split the data into this amount of smaller
        tables for distribution to worker nodes in distributed computing
        environments.

    Returns
    -------

    bardi.data.data_handlers.Dataset
        bardi Dataset object with the data attribute referencing the data that
        was supplied after conversion to a PyArrow Table.

    Raises
    ------

    ValueError if the supplied file path does not contain a filetype of an
        accepted format.
    """

    format = format.lower()
    accepted_formats = ["parquet", "ipc", "arrow", "feather", "csv", "orc"]

    if format not in accepted_formats:
        raise ValueError(
            "The format given is not supported. " f"Expected one of {accepted_formats}"
        )
    else:
        # Create a PyArrow Dataset object from the file(s)
        # This step doesn't actually read the data yet, but just references it
        data = ds.dataset(source=source, format=format, *args, **kwargs)
        table = data.to_table()

        # Determine the starting number of rows in the data table
        row_count = table.num_rows

        # Create a bardi Dataset object which will reference the data and
        # additional metadata
        dataset_obj = Dataset()
        dataset_obj.origin_row_count = row_count
        dataset_obj.origin_file_path = source
        dataset_obj.origin_format = format

        if min_batches:
            # If a min_batches is specified, data from the file(s) are split
            # into a list of smaller PyArrow Tables
            tables = chunk_pyarrow_table(
                data=table, row_count=row_count, min_batches=min_batches
            )

            # Setting the bardi Dataset object's data references to the list
            # of PyArrow Tables
            dataset_obj.data = tables

        else:
            # If min_batches is not specified, all of the data is returned in
            # a single PyArrow Table
            dataset_obj.data = table

        return dataset_obj


def from_duckdb(path: str, query: str, min_batches: int = None) -> Dataset:
    """Create a bardi Dataset object using data returned from
    a custom query on a DuckDB database

    Parameters
    ----------

    path : str
        A filepath to the DuckDB database file
    query : str
        A valid SQL query adhering to DuckDB syntax specifications
    min_batches : int
        An integer number used to split the data into this amount of smaller
        tables for distribution to worker nodes.
        Number will probably align with the number of worker nodes.

    Returns
    -------

    bardi.data.data_handlers.Dataset
        bardi Dataset object with the data attribute referencing the data that
        was supplied after conversion to a PyArrow Table.
    """

    # Create a read-only connection to DuckDB database file and execute the
    # query returning a PyArrow Table
    conn = duckdb.connect(path, read_only=True)
    table = conn.execute(query).fetch_arrow_table()
    row_count = table.num_rows

    # Create a bardi Dataset object which will reference the data and
    # additional metadata
    dataset_obj = Dataset()
    dataset_obj.origin_file_path = path
    dataset_obj.origin_query = query
    dataset_obj.origin_format = "duckdb"
    dataset_obj.origin_row_count = row_count

    if min_batches:
        # If a min_batches is specified, data from the file(s) are split into a
        # list of smaller PyArrow Tables
        tables = chunk_pyarrow_table(
            data=table, row_count=row_count, min_batches=min_batches
        )

        # Setting the bardi Dataset object's data references to the list of
        # PyArrow Tables
        dataset_obj.data = tables

    else:
        # If min_batches is not specified, all of the data is returned in a
        # single PyArrow Table
        dataset_obj.data = table

    return dataset_obj


def from_pandas(df: pd.DataFrame, min_batches: int = None) -> Dataset:
    """Create a bardi dataset object from a Pandas DataFrame using the PyArrow
    function

    Parameters
    ----------

    df : Pandas DataFrame
        A Pandas DataFrame containing data intended to be passed into a bardi
        pipeline
    distributed : bool
        A flag which prompts splitting data into smaller chunks to prepare for
        later distribution to worker nodes.
        Also used to set a flag in the bardi Dataset object to direct future
        operations to be performed in a distributed manner.
    min_batches : int
        An integer number used to split the data into this amount of smaller
        tables for distribution to worker nodes.
        Number will probably align with the number of worker nodes.

    Returns
    -------

    bardi.data.data_handlers.Dataset
        bardi Dataset object with the data attribute referencing the data that
        was supplied after conversion to a PyArrow Table.
    """
    # Convert the Pandas DataFrame to a PyArrow Table
    table = pa.Table.from_pandas(df)
    row_count = table.num_rows

    # Create a bardi Dataset object which will reference the data and
    # additional metadata
    dataset_obj = Dataset()
    dataset_obj.origin_format = "pandas"
    dataset_obj.origin_row_count = row_count

    if min_batches:
        # If a min_batches is specified, data from the file(s) are split into
        # a list of smaller PyArrow Tables
        tables = chunk_pyarrow_table(
            data=table, row_count=row_count, min_batches=min_batches
        )

        # Setting the bardi Dataset object's data references to the list of
        # PyArrow Tables
        dataset_obj.data = tables

    else:
        # If min_batches is not specified, all of the data is returned in a
        # single PyArrow Table
        dataset_obj.data = table

    return dataset_obj


def from_pyarrow(table: pa.Table, min_batches: int = None) -> Dataset:
    """Create a bardi dataset object from an existing PyArrow Table

    Parameters
    ----------

    table : PyArrow Table
        A PyArrow Table containing data intended to be passed into a bardi
        pipeline
    min_batches : int
        An integer number used to split the data into this amount of smaller
        tables for distribution to worker nodes.
        Number will probably align with the number of worker nodes.

    Returns
    -------

    bardi.data.data_handlers.Dataset
        bardi Dataset object with the data attribute referencing the data that
        was supplied after conversion to a PyArrow Table.
    """

    row_count = table.num_rows

    # Create a bardi Dataset object which will reference the data and
    # additional metadata
    dataset_obj = Dataset()
    dataset_obj.origin_format = "pyarrow"
    dataset_obj.origin_row_count = row_count

    if min_batches:
        # If a min_batches is specified, data from the file(s) are split into a
        # list of smaller PyArrow Tables
        tables = chunk_pyarrow_table(
            data=table, row_count=row_count, min_batches=min_batches
        )

        # Setting the bardi Dataset object's data references to the list of
        # PyArrow Tables
        dataset_obj.data = tables

    else:
        # If min_batches is not specified, all of the data is returned in a
        # single PyArrow Table
        dataset_obj.data = table

    return dataset_obj


def from_json(json_data: Union[str, dict, List(dict)]) -> Dataset:
    """Create a bardi Dataset object from JSON data

    Parameters
    ----------

    json_data : str
        An object of name/value pairs. Names will become columns in the PyArrow
        Table.

    Returns
    -------
    bardi.data.data_handlers.Dataset
        bardi Dataset object with the data attribute referencing the data that
        was supplied after conversion to a PyArrow Table.
    """

    # Convert the JSON object into a dictionary and then load that as a single
    # row in a PyArrow Table
    if isinstance(json_data, str):
        records = [json.loads(json_data)]
    elif isinstance(json_data, dict):
        # If the JSON string was already deserialized upstream
        records = [json_data]
    elif isinstance(json_data, list):
        records = json_data
    else:
        raise TypeError(
            "json data must be provided as a serialized json str, "
            "a deserialized dictionary, or a list of dictionaries."
        )

    table = pa.Table.from_pylist(records)
    row_count = table.num_rows

    # Create a bardi Dataset object which will reference the data and
    # additional metadata
    dataset_obj = Dataset()
    dataset_obj.data = table
    dataset_obj.origin_format = "json"
    dataset_obj.origin_row_count = row_count

    return dataset_obj


# ======= Writers ========


def to_pandas(table: pa.Table) -> pd.DataFrame:
    """Return data as a pandas DataFrame
    
    Parameters
    ----------

    table : PyArrow.Table
        Table of data you want to convert to a Pandas DataFrame
    
    Returns
    -------

    pandas.DataFrame
        The same data as the input table, converted into a DataFrame
    """
    df = table.to_pandas()
    return df


def to_polars(table: pa.Table) -> pl.DataFrame:
    """Return data as a polars DataFrame
    
    Parameters
    ----------

    table : PyArrow.Table
        Table of data you want to convert to a Pandas DataFrame
    
    Returns
    -------

    polars.DataFrame
        The same data as the input table, converted into a DataFrame
    """
    df = pl.from_arrow(table)
    return df


def write_file(data: pa.Table, path: str, format: str, *args, **kwargs) -> None:
    """Write data to a file

    Note
    ----

    Only a subset of possible arguments are presented here. Additional arguments
    can be passed for specific file types.
    Reference PyArrow documentation for additional arguments.

    Parameters
    ----------

    data : PyArrow Table
    path : str
        path in filesystem where data is to be written
    format : str
        filetype in "parquet", "feather", "csv", "orc", "json", "npy"
    """

    format = format.lower()
    accepted_formats = ["parquet", "feather", "csv", "orc", "json", "npy"]

    if format not in accepted_formats:
        raise ValueError(
            "The format given is not supported. " f"Expected one of {accepted_formats}"
        )
    else:
        if format == "orc":
            orc.write_table(data, path, *args, **kwargs)
        elif format == "parquet":
            pq.write_table(data, path, *args, **kwargs)
        elif format == "feather":
            feather.write_feather(data, path, *args, **kwargs)
        elif format == "csv":
            # Convert list columns to strings
            csv_df = pl.from_arrow(data)
            schema = csv_df.schema
            csv_df = csv_df.with_columns(
                [
                    pl.format(
                        "[{}]", pl.col(field).cast(pl.List(pl.Utf8)).list.join(", ")
                    ).alias(field)
                    for field in schema.keys()
                    if schema[field] == pl.List()
                ]
            )
            data = csv_df.to_arrow()
            csv.write_csv(data, path, *args, **kwargs)
        elif format == "json":
            json_object = json.dumps(data, indent=4)
            with open(path, "w") as f:
                f.write(json_object)
        elif format == "npy":
            with open(path, "wb") as f:
                np.save(file=f, arr=data, *args, **kwargs)
