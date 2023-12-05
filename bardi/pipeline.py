import os
import tracemalloc
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import List, Literal, Tuple, TypedDict, Union

import pyarrow as pa

from bardi.data import data_handlers
from bardi.data.data_handlers import Dataset


class DataWriteConfig(TypedDict):
    data_format: str
    data_format_args: Union[dict, None]


class Step(metaclass=ABCMeta):
    """Pipeline step abstract base class

    To be used as a blueprint for creating new steps for a data
    pre-processing pipelines.
    """
    @abstractmethod
    def __init__(self):
        self._data_write_config: DataWriteConfig = {
            "data_format": 'parquet',
            "data_format_args": {"compression": 'snappy',
                                 "use_dictionary": False}
        }  # Default write config

    @abstractmethod
    def run(self, data: pa.Table, artifacts: dict,
            ) -> Tuple[Union[pa.Table, None], Union[dict, None]]:
        """Implement a run method that will be called by the
        pipeline's run_pipeline method.

        Args:
            - Expect to receive a PyArrow table of data
            - Expect to receive a dict of artifacts
              - artifacts can be ignored if values are not needed from it
        Returns:
            - If the method performs a transformation of data then return
            the data as a PyArrow table. This will replace the pipeline
            object's processed_data attribute.
            - If the method uses data to create new artifacts, return a
            dictionary with keys corresponding to the name of the artifact
            and the values being the artifact's data or value
        """
        pass

    def set_write_config(self, data_config: DataWriteConfig):
        """Provide a method to customize the write configuration
        used by the step's write_outputs method
        """
        self._data_write_config = data_config

    def get_parameters(self) -> dict:
        """Implement a get_parameters method that will be called by
        the pipeline's get_parameters method

        In many cases, this can be implemented with the following
        built-in Python vars function:

        def get_parameters(self) -> dict:
            params = vars(self)
            return params

        Args:
            None
        Returns:
            - A dictionary of all of the configuration details of the
            object
        """
        params = vars(self).copy()
        return params

    def write_data(self, write_path: str, data: Union[pa.Table, None],
                   data_filename: Union[str, None] = None):
        """Default implementation of the write_data method.

        Reuse existing write patterns that exist in the data handlers.

        Args:
            data: PyArrow Table of data
            write_path : str
                a directory where a file should be persisted if write_data is
                set to True
            data_filename: name of the file where final data output
                           is saved

        Returns:
            None
        """
        # Add the filename with the appropriate filetype to the write path
        if not data_filename:
            data_filename = f'{self.__class__.__name__}Data'
        file_path = os.path.join(
            write_path,
            (f'{data_filename}.{self._data_write_config["data_format"]}')
        )
        # Call the data handler write_file function
        data_handlers.write_file(
            data=data,
            path=file_path,
            format=self._data_write_config['data_format'],
            **self._data_write_config['data_format_args'])

    def write_artifacts(self, write_path: str,
                        artifacts: Union[dict, None]):
        """Default implementation of the write_artifacts method.
        Since only some steps require this methods - the default
        behaviour is 'pass'.

        Reuse existing write patterns that exist in the data handlers.

        Args:
            write_path : str
                a directory where a file should be persisted if write_data is
                set to True
            artifacts: Union[dict,None]


        Returns:
            None

        """
        pass


class Pipeline:
    """Organizes Steps into a pipeline to operate on data from a Dataset"""

    def __init__(self,
                 dataset: Dataset = None,
                 write_path: str = None,
                 write_outputs: Literal['pipeline-outputs',
                                        'debug',
                                        False] = 'pipeline-outputs',
                 data_write_config: DataWriteConfig = None,
                 data_filename: str = "bardi_processed_data"):

        # Ref a bardi dataset object that pipeline operates on
        self.dataset: Dataset = dataset

        # Pipeline configuration
        self.steps: List[Step] = []
        self.num_steps = 0
        self.write_outputs = write_outputs
        self.write_path = write_path
        self.data_filename = data_filename

        if data_write_config:
            self.data_write_config = data_write_config
        else:
            self.data_write_config = {
                "data_format": 'parquet',
                "data_format_args": {"compression": 'snappy',
                                     "use_dictionary": False}
            }  # Default config, but can be overwritten
        if write_outputs:
            if not os.path.isdir(self.write_path):
                raise ValueError('Pipeline write_outputs was configured to '
                                 'write, however a write_path was not '
                                 'correctly specified. Please provide a '
                                 'path to a directory to write files into.')

        # Tracking
        self.artifacts = {}
        self.performance = {}

    def add_step(self, step: Step) -> None:
        """Adds a step object to list of pipeline execution steps.

        Also overwrites the step's data write configuration with a consistent
        pipeline data write configuration.
        """
        if isinstance(step, Step):
            step.set_write_config(data_config=self.data_write_config)
            self.steps.append(step)
            self.num_steps += 1
        else:
            raise TypeError('Only objects of type Step may be added to '
                            'a bardi pipeline.')

    def run_pipeline(self) -> None:
        """Calls the run and write_outputs method for each respective step
        object added to the pipeline"""

        if isinstance(self.dataset, Dataset):
            self.processed_data = self.dataset.data
        else:
            raise TypeError("Pipeline's Dataset must reference a bardi "
                            "Dataset object. Please utilize bardi's data_"
                            "handlers to create a bardi Dataset object from "
                            "your data.")

        # For each step of the pipeline, call its run method
        pipeline_start_time = datetime.now()
        for step_position, step in enumerate(self.steps, start=1):

            # Call the run method and time the execution
            step_start_time = datetime.now()
            tracemalloc.start()
            results = step.run(data=self.processed_data,
                               artifacts=self.artifacts)
            step_max_mem = tracemalloc.get_traced_memory()[1] / 1000000
            tracemalloc.stop()
            step_end_time = datetime.now()
            step_run_time = step_end_time - step_start_time

            # Record the performance
            if self.write_outputs == 'debug':
                print(f'{str(type(step))} run time: {step_run_time}')
                print(f'{str(type(step))} max memory (MB): {step_max_mem}')
            self.performance[str(type(step))] = {
                "time": str(step_run_time),
                "memory (MB)": str(step_max_mem)
            }

            if isinstance(results, tuple):
                if isinstance(results[0], pa.Table):
                    # Set the pipeline's data state to the newest data results
                    self.processed_data = results[0]
                    if self.write_outputs == 'debug':
                        step.write_data(write_path=self.write_path,
                                        data=self.processed_data)
                    elif (step_position == self.num_steps
                          and self.write_outputs == 'pipeline-outputs'):
                        step.write_data(write_path=self.write_path,
                                        data=self.processed_data,
                                        data_filename=self.data_filename)
                if isinstance(results[1], dict):
                    # If artifacts were returned by the step, add them to
                    # the pipeline's total set of artifacts
                    self.artifacts = {**self.artifacts, **results[1]}

                    if self.write_outputs:
                        step.write_artifacts(write_path=self.write_path,
                                             artifacts=self.artifacts)
            else:
                raise TypeError("Pipeline expected step to return a tuple of "
                                "PyArrow Table of data and a dictionary of "
                                "artifacts. If the step doesn't return one of "
                                "these, that position in the tuple can be "
                                "empty, but it still needs to return a tuple.")

        # Record the total pipeline performance
        pipeline_end_time = datetime.now()
        pipeline_run_time = pipeline_end_time - pipeline_start_time
        if self.write_outputs == 'debug':
            print(f'Pipeline run time: {pipeline_run_time}')
        self.performance[str(type(self))] = str(pipeline_run_time)

    def get_parameters(self, condensed: bool = True) -> dict:
        """Returns the params of the pipeline's dataset and params of each step

        Keyword Arguments:
            condensed : bool
                If True, the step configuration dictionary will exclude any
                attributes set to None or False

        Returns:
            dict of params used to configure each pipeline step object
        """

        pipeline_params = {
            "dataset": {},
            "steps": {},
            "performance": {}
        }

        # Get parameters for the pipeline's dataset
        if self.dataset:
            dataset_params = self.dataset.get_parameters()
            if condensed:
                condensed_params = {}
                for attribute in dataset_params.keys():
                    if dataset_params[attribute] not in [False, None]:
                        condensed_params[attribute] = dataset_params[attribute]
                pipeline_params["dataset"][str(type(self.dataset))] = condensed_params
            else:
                pipeline_params["dataset"][str(type(self.dataset))] = dataset_params

        # Get parameters for each step in the pipeline
        for i, step in enumerate(self.steps):
            step_params = step.get_parameters()

            if condensed:
                condensed_params = {}
                for attribute in step_params.keys():
                    if step_params[attribute] not in [False, None]:
                        condensed_params[attribute] = step_params[attribute]
                pipeline_params["steps"][str(type(step))] = condensed_params
            else:
                pipeline_params["steps"][str(type(step))] = step_params

        # Get the pipeline performance
        pipeline_params["performance"] = self.performance

        return pipeline_params
