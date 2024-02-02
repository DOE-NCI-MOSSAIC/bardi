"""Segment the dataset into splits, such as `test, train, and val`"""

from abc import abstractmethod
from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
import polars as pl
import pyarrow as pa

from bardi.nlp_engineering.utils.validations import validate_pyarrow_table
from bardi.pipeline import DataWriteConfig, Step

MapSplit = NamedTuple(
    "MapSplit", [("unique_record_cols", List[str]), ("split_mapping", Dict[str, str])]
)
MapSplit.__doc__ = (
    """Specify the requirements for splitting data exactly in line with 
    an existing data split
    """
)
MapSplit.unique_record_cols.__doc__ = (
    """List of column names of which the combination forms a unique identifier in the
    dataset. This can be a single column name if that creates a unique identifier,
    but oftentimes in datasets a combination of fields are required.

    Note: This set of columns MUST create a unique record or the program will crash.
    """
)
MapSplit.split_mapping.__doc__ = (
    """`Only used for map split type.`
    A dictionary mapping where the keys are the hash of the concatenated values from
    unique_record_cols, or represented by the following pseudocode, ::
        hash(concat(*unique_record_cols))
    The values are the corresponding split value (train, test, val) or (fold1, fold2, fold3), etc.
    """
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
NewSplit.__doc__ = (
    "Specify the requirements for splitting data with a new split from scratch."
)
NewSplit.split_proportions.__doc__ = (
    """`Only used for new split type.`
    Mapping of split names to split proportions. i.e., ::
        {'train': 0.75, 'test': 0.15, 'val': 0.15}
        {'fold1': 0.25, 'fold2': 0.25, 'fold3': 0.25, 'fold4': 0.25}
    Note: values must add to 1.0.
    """
)
NewSplit.unique_record_cols.__doc__ = (
    """List of column names of which the combination forms a unique identifier in the
    dataset. This can be a single column name if that creates a unique identifier,
    but oftentimes in datasets a combination of fields are required.

    Note: This set of columns MUST create a unique record or the program will crash.
    """
)
NewSplit.group_cols.__doc__ = (
    """List of column names that form a 'group' that you would like to keep in discrete
    splits. E.x., if you had multiple medical notes for a single patient,
    you may desire that all notes for a single patient end up in the same split
    to prevent potential information leakage. In this case you would provide
    something like a patient_id.
    """
)
NewSplit.label_cols.__doc__ = (
    """List of column names containing labels. Efforts are made to balance label
    distribution across splits, but this is not guaranteed.
    """
)
NewSplit.random_seed.__doc__ = (
    "Required for reproducibility. If you have no preference, try on 42 for size."
)


class Splitter(Step):
    """The splitter adds a 'split' column to the data assigning
    each record to a particular split for downstream model training.

    Note
    ----

    Avoid the direct instantiation of the Splitter class
    and instead instantiate one of the child classes.
    
    Attributes
    ----------

    split_method : Union[MapSplit, NewSplit]
        A named tuple of either MapSplit type or NewSplit
        type. Each contains a different set of values
        used to create the splitter depending upon split type
    split_type : str
        The type of split to be performed:
            * new - create a new random data split
            * map - reproduce an existing data split by mapping unique IDs
    unique_record_cols : List[str]
        List of column names of which the combination forms a unique identifier in the
        dataset. This can be a single column name if that creates a unique identifier,
        but oftentimes in datasets a combination of fields are required.

        Note: This set of columns MUST create a unique record or the program will crash.
    split_mapping : dict[str, str]
        `Only used for map split type.`
        A dictionary mapping where the keys are the hash of the concatenated values from
        unique_record_cols, or represented by the following pseudocode, ::
            hash(concat(*unique_record_cols))
        The values are the corresponding split value (train, test, val) or (fold1, fold2, fold3), etc.
    split_proportions : dict[str, float]
        `Only used for new split type.`
        Mapping of split names to split proportions. i.e., ::
            {'train': 0.75, 'test': 0.15, 'val': 0.15}
            {'fold1': 0.25, 'fold2': 0.25, 'fold3': 0.25, 'fold4': 0.25}
        Note: values must add to 1.0.
    num_splits : int
        The number of splits contained in `split_proportions`.
    group_cols : List[str]
        List of column names that form a 'group' that you would like to keep in discrete
        splits. E.x., if you had multiple medical notes for a single patient,
        you may desire that all notes for a single patient end up in the same split
        to prevent potential information leakage. In this case you would provide
        something like a patient_id.
    label_cols : List[str]
        List of column names containing labels. Efforts are made to balance label
        distribution across splits, but this is not guaranteed.
    random_seed: int
        Required for reproducibility. If you have no preference, try on `42` for size.
    """

    def __init__(
        self,
        split_method: Union[MapSplit, NewSplit],
    ):
        """Constructor method
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
            "data_format_args": {"compression": "snappy", "use_dictionary": False},
        }  # Default write config

    @abstractmethod
    def run(self):
        """Abstract method
        """
        pass


class CPUSplitter(Splitter):
    """The splitter adds a 'split' column to the data assigning
    each record to a particular split for downstream model training.

    Two split types are available - creating a new random split
    from scratch and assigning previously created splits. This
    second option is helpful when running comparisons with
    other methods of data processing ensuring that splits are
    exactly the same.

    Note
    ----
    This implementation of the LabelProcessor is specific for CPU computation.
    
    To create a splitter, pass the appropriate set of parameters through a defined
    NamedTuple for the type of split you want to create. i.e., ::

        CPUSplitter(split_method=NewSplit(
            split_proportions={
                'train': 0.75,
                'test': 0.15,
                'val': 0.15
            },
            unique_record_cols=[
                'document_id'
            ],
            group_cols=[
                'patient_id_number',
                'registry'
            ],
            labels_cols=[
                'reportability'
            ],
            random_seed=42
        ))
    
    Splitter Method Named Tuples:
    
        *   :mod:`bardi.nlp_engineering.splitter.NewSplit`
        *   :mod:`bardi.nlp_engineering.splitter.MapSplit`

    Attributes
    ----------
    split_method : Union[MapSplit, NewSplit]
        A named tuple of either MapSplit type or NewSplit
        type. Each contains a different set of values
        used to create the splitter depending upon split type
    split_type : str
        The type of split to be performed:
            * new - create a new random data split
            * map - reproduce an existing data split by mapping unique IDs
    unique_record_cols : List[str]
        List of column names of which the combination forms a unique identifier in the
        dataset. This can be a single column name if that creates a unique identifier,
        but oftentimes in datasets a combination of fields are required.

        Note: This set of columns MUST create a unique record or the program will crash.
    split_mapping : dict[str, str]
        `Only used for map split type.`
        A dictionary mapping where the keys are the hash of the concatenated values from
        unique_record_cols, or represented by the following pseudocode, ::
            hash(concat(*unique_record_cols))
        The values are the corresponding split value (train, test, val) or (fold1, fold2, fold3), etc.
    split_proportions : dict[str, float]
        `Only used for new split type.`
        Mapping of split names to split proportions. i.e., ::
            {'train': 0.75, 'test': 0.15, 'val': 0.15}
            {'fold1': 0.25, 'fold2': 0.25, 'fold3': 0.25, 'fold4': 0.25}
        Note: values must add to 1.0.
    num_splits : int
        The number of splits contained in `split_proportions`.
    group_cols : List[str]
        List of column names that form a 'group' that you would like to keep in discrete
        splits. E.x., if you had multiple medical notes for a single patient,
        you may desire that all notes for a single patient end up in the same split
        to prevent potential information leakage. In this case you would provide
        something like a patient_id.
    label_cols : List[str]
        List of column names containing labels. Efforts are made to balance label
        distribution across splits, but this is not guaranteed.
    random_seed: int
        Required for reproducibility. If you have no preference, try on `42` for size.
    """

    def __init__(self, *args, **kwargs):
        """Constructor method
        """
        super().__init__(*args, **kwargs)

    def run(self, data: pa.Table, artifacts: dict = None) -> Tuple[pa.Table, dict]:
        """Runs a splitter using CPU computation based on the configuration
        used to create the object of the CPUSplitter class

        Parameters
        ----------

        data : PyArrow Table
            The data to be split
        artifacts : dict
            artifacts are not consumed in this run method, but must be
            received to operate correctly in the pipeline run method

        Returns
        -------

        Tuple(PyArrow.Table, dict)
            A tuple with:
                *   the first element holding a PyArrow Table of data
                    including the new split column
                *   the second element of the tuple intended for
                    artifacts is None

        Raises
        ------

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
            distinct_groups = df.select([*self.group_cols]).unique(maintain_order=True)
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

            # Add split labels as a column to distince groups
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
