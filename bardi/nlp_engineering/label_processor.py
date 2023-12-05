import os
from abc import abstractmethod
from typing import List, Tuple, TypedDict, Union

import polars as pl
import pyarrow as pa

from bardi.data import data_handlers
from bardi.nlp_engineering.utils.validations import validate_pyarrow_table
from bardi.pipeline import DataWriteConfig, Step


class LabelProcessorArtifactsWriteConfig(TypedDict):
    """Indicates the keys and data types expected in an
    artifacts write config dict for the label processor
    if overwriting the default configuration."""
    id_to_label_format: str
    id_to_label_args: Union[dict, None]


class LabelProcessor(Step):
    """The label processor creates and maps a label vocab

    Avoid the direct instantiation of the LabelProcessor class
    and instead instantiate one of the child classes.

    Attributes:
        field : str
            the name of a label column of which the values
            are used to generate a standardized mapping and
            then that mapping is applied to the column
        method : str
            currently only a default 'unique' method is
            supported which maps each unique value in the
            column to an id
        mapping : dict
            mapping dict of the form {label: id} used to convert
            labels in the column to ids
        id_to_label : dict
            the reverse of mapping. of the form {id: label} used
            downstream to map the ids back to the original label
            values

    Methods:
        run
        get_parameters
    """
    def __init__(self,
                 fields: Union[str, List[str]],
                 method: str = 'unique'
                 ):
        """Create an object of the LabelProcessor class

        Keyword Arguments:
            field : str
                the name of a label column of which the values
                are used to generate a standardized mapping and
                then that mapping is applied to the column
            method : str
                currently only a default 'unique' method is
                supported which maps each unique value in the
                column to an id
        """
        self.fields = fields
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = fields
        self.method = method
        self.mapping = {}
        self.id_to_label = {}
        self._data_write_config: DataWriteConfig = {
            "data_format": 'parquet',
            "data_format_args": {"compression": 'snappy',
                                 "use_dictionary": False}
        }
        self._artifacts_write_config: LabelProcessorArtifactsWriteConfig = {
            "id_to_label_format": 'json',
            "id_to_label_format_args": {}
        }

    def set_write_config(self, data_config: DataWriteConfig = None,
                         artifacts_config: LabelProcessorArtifactsWriteConfig = None):
        """Overwrite the default file writing configurations"""
        if data_config:
            self._data_write_config = data_config
        if artifacts_config:
            self._artifacts_write_config = artifacts_config

    @abstractmethod
    def run(self):
        """to be implemented in child class"""
        pass


class CPULabelProcessor(LabelProcessor):
    """Implementation of the LabelProcessor for CPU computation

    Inherited Attributes:
        fields : str or List[str]
        method : str
        mapping : dict
        id_to_label : dict

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

    def run(self, data: pa.Table, artifacts: dict = None) -> Tuple[pa.Table, dict]:
        """Run a label processor using CPU computation

        Keyword Arguments:
            data : PyArrow Table
                The data to be processed. The data must contain the
                column specified by 'field' at object creation
            artifacts : dict
                artifacts are not used in this run method, but must be received
                to operate correctly in the pipeline run method

        Returns:
            Tuple[pyarrow.Table, dict]
                The first position is a pyarrow.Table of processed data
                The second position is a dictionary of artifacts. The dict will
                contain a key for "id_to_label".

        Raises
            NotImplementedError
                A value other than 'unique' was provided for the
                label processor's method
            TypeError
                The run method was not supplied a PyArrow Table
        """
        # Perform validations
        validate_pyarrow_table(data=data)

        # Confirm that the method is passed a PyArrow Table
        produced_artifacts = {}
        for field in self.fields:
            if self.method == 'unique':
                try:
                    # Get the unique labels
                    vals = (pl.from_arrow(data)
                            .get_column(field)
                            .unique()
                            .sort()
                            ).to_list()

                    # Create a mapping for the labels
                    field_mapping = {label: id
                                     for id, label in enumerate(vals)}
                    # id_to_label is flipped vs what is needed for mapping
                    field_id_to_label = {str(id): str(label)
                                         for label, id
                                         in field_mapping.items()}
                    self.id_to_label[field] = field_id_to_label

                    df = (pl.from_arrow(data)
                            .with_columns(pl.col(field)
                                          .map_dict(field_mapping,
                                                    default=None)))

                except Exception as e:
                    print(f'An error occured: {e} '
                          'Processing without creating the label '
                          f'mapping for {field}')
                    artifact = {"id_to_label": {field: None}}
                    return (data, artifact)

            else:
                raise NotImplementedError(
                    "Only unique mapping is currently supported"
                )

            data = df.to_arrow()

        # Set up the artifacts dict to return
        produced_artifacts = {
            'id_to_label': self.id_to_label
        }

        return (data, produced_artifacts)

    def write_artifacts(self, write_path: str,
                        artifacts: Union[dict, None]) -> None:
        """Write the outputs produced by the label_processor

        Keyword Arguments:
            write_path : str
                Path is a directory where files will be written
            artifacts : Union[dict, None]
                Artifacts is a dictionary of artifacts produced in this step.
                Expected key is: "id_to_label"

        Returns:
            None
        """
        id_to_token = artifacts['id_to_label']
        # Add the filename with the appropriate filetype to the write path
        id_to_token_path = os.path.join(
            write_path,
            (f'id_to_label'
                f'.{self._artifacts_write_config["id_to_label_format"]}')
        )
        # Call the data handler write_file function
        data_handlers.write_file(
            data=id_to_token,
            path=id_to_token_path,
            format=self._artifacts_write_config['id_to_label_format'],
            **self._artifacts_write_config['id_to_label_format_args']
        )

    def get_parameters(self) -> dict:
        """Retrive the label processor object configuration

        Does not return the mapping (vocab), but does return the id_to_label
        dict. This is because the mapping is just the reverse of id_to_label.

        Returns:
            a dictionary representation of the splitter object's attributes
        """
        params = vars(self).copy()
        params.pop('mapping')
        return params
