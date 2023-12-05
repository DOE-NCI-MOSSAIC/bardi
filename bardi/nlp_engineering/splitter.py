"""Segment the dataset into splits, such as test, train, and val"""

from abc import abstractmethod
from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
import polars as pl
import pyarrow as pa

from bardi.nlp_engineering.utils.validations import validate_pyarrow_table
from bardi.pipeline import DataWriteConfig, Step

MapSplit = NamedTuple(
    "MapSplit", [("unique_record_cols", List[str]),
                 ("split_mapping", Dict[str, str])
                 ]
)
NewSplit = NamedTuple(
    "NewSplit",
    [
        ("split_proportions", Dict[str, float]),
        ("unique_record_cols", List[str]),
        ("group_cols", List[str]),
        ("label_cols", List[str]),
        ("random_seed", int),
    ],
)


class Splitter(Step):
    """The splitter adds a 'split' column to the data assigning
    each record to a particular split for downstream model training.

    Two split types are available - creating a new random split
    from scratch and assigning previously created split. This
    second option is helpful when running comparisons with
    other methods of data processing ensuring that splits are
    exactly the same.

    Attributes:
        split_type : str
        unique_record_cols : List[str]
        split_mapping : dict[str, str]
        split_proportions : dict[str, float]
        num_splits : int
        group_cols : List[str]
        label_cols : List[str]
        random_seed: int

    Methods:
        run
        get_parameters
    """

    def __init__(
        self,
        split_method: Union[MapSplit, NewSplit],
    ):
        """Create an object of the Splitter class

        Keyword Arguments:
            split_method : Union[MapSplit, NewSplit]
                A named tuple of either MapSplit type or NewSplit
                type. Each contains a different set of values
                used to create the splitter depending upon split type
        """
        if isinstance(split_method, MapSplit):
            self.split_type = "map"
            self.unique_record_cols = split_method.unique_record_cols
            self.split_mapping = split_method.split_mapping
        elif isinstance(split_method, NewSplit):
            self.split_type = "new"
            self.split_proportions = split_method.split_proportions
            self.num_splits = len(self.split_proportions)
            self.unique_record_cols = split_method.unique_record_cols
            self.group_cols = split_method.group_cols
            self.label_cols = split_method.label_cols
            self.random_seed = split_method.random_seed
        self._data_write_config: DataWriteConfig = {
            "data_format": "parquet",
            "data_format_args": {"compression": "snappy",
                                 "use_dictionary": False},
        }  # Default write config

    @abstractmethod
    def run(self):
        """to be implemented in child class"""
        pass


class CPUSplitter(Splitter):
    """Implementation of the Splitter for CPU computation

    Inherited Attributes:
        split_type : str
        unique_record_cols : List[str]
        split_mapping : dict[str, str]
        split_proportions : dict[str, float]
        num_splits : int
        group_cols : List[str]
        label_cols : List[str]
        random_seed : int

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
            artifacts: dict = None) -> Tuple[pa.Table, dict]:
        """Runs a splitter using CPU computation based on the configuration
        used to create the object of the CPUSplitter class

        Keyword Arguments:
            data : PyArrow Table
                The data to be split
            artifacts : dict
                artifacts are not consumed in this run method, but must be
                received to operate correctly in the pipeline run method

        Returns:
            Tuple(PyArrow Table, dict)
                A tuple with:
                 - the first element holding a PyArrow Table of data
                 including the new split column
                 - the second element of the tuple intended for
                 artifacts is None

        Raises
            TypeError
                The run method was not supplied a PyArrow Table
        """
        # Perform validations
        validate_pyarrow_table(data=data)

        df = pl.from_arrow(data)

        # Mapping from an existing data split - Good for comparisons
        if self.split_type == "map":
            # Apply an existing mapping of id -> split

            # Combine and hash the columns that form a unique/distinct
            # record and map a dictionary set up as:
            # {id: split} where id is also a combined and hashed
            # combination of the columns forming a unique record
            df = (
                df.with_columns(
                    [
                        pl.concat_str([*self.unique_record_cols])
                        .hash()
                        .alias("composite_record_id")
                    ]
                )
                .with_columns(
                    [
                        pl.col("composite_record_id")
                        .cast(pl.Utf8())
                        .map_dict(self.split_mapping)
                        .alias("split")
                    ]
                )
                .drop("composite_record_id")
            )

        elif self.split_type == "new":
            # Create a new split of the data if an existing desired
            # split does not yet exist

            # Get the set of distinct groups from the overall data
            distinct_groups = (df.select([*self.group_cols])
                                 .unique(maintain_order=True))
            group_count = distinct_groups.height
            # Generate a permutation of indices of distinct groups.
            np.random.seed(self.random_seed)
            permuted_indices = np.random.permutation(group_count)

            # The splits assignment has to be done with a number
            # so a mapping must be created from a number to the
            # desired split name
            split_name_maps = {
                i: key for i, key in enumerate(self.split_proportions.keys())
            }

            # Create a numpy array to hold the split labels (digits)
            split_labels = np.empty(group_count, dtype=int)
            start, end = 0, 0
            # Loop over the split proportion and assign digit
            # labels to the corresponding indices. If 15% of
            # data should be in a given split then we take a next
            # slice of permutated indices of that size and assign to them
            # a given digit label.
            for i, split_proportion in enumerate(self.split_proportions.items()):
                end = start + int(split_proportion[1] * group_count)
                split_labels[permuted_indices[start:end]] = i
                start = end
            # In case some of incides are left out due to divisibility.
            split_labels[permuted_indices[end:]] = i
            split_labels = pl.Series(split_labels)

            # Add split lables as a column to distince groups
            # mapped to assigned value back to the name of
            # the split
            splits = distinct_groups.with_columns(
                split_labels.map_dict(split_name_maps).alias("split")
            )

            # Join the split set of distinct groups back to the rest
            # of the data. If a group had more than one record in the
            # dataset, this join will ensure those records are mapped
            # to the same split as the others in the group
            df = df.join(splits, on=[*self.group_cols], how="left")

        data = df.to_arrow()

        return (data, None)
