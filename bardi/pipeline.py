"""Defines a pipeline and a framework for steps to run in it"""

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
    """Blueprint for creating new steps for data pre-processing pipelines"""

    @abstractmethod
    def __init__(self):
        """Constructor method
        """
        self._data_write_config: DataWriteConfig = {
            "data_format": "parquet",
            "data_format_args": {"compression": "snappy", "use_dictionary": False},
        }  # Default write config

    @abstractmethod
    def run(
        self,
        data: pa.Table,
        artifacts: dict,
    ) -> Tuple[Union[pa.Table, None], Union[dict, None]]:
        """Implement a run method in the step that will be called by the
        pipeline's run_pipeline method.

        Parameters
        ----------

        data : PyArrow.Table
            Expect to receive a PyArrow table of data
        artifacts : dict
            Expect to receive a dict of artifacts from preceding steps,
            but artifacts can be ignored in the method if values are not needed from it

        Returns
        -------
        Tuple[Union[pa.Table, None], Union[dict, None]]
            If the method performs a transformation of data then return
            the data as a PyArrow table. This will replace the pipeline
            object's processed_data attribute. If the method creates new artifacts
            besides the data, return a dictionary with keys corresponding to the name of
            the artifact and the values being the artifact's data or value
        """
        pass

    def set_write_config(self, data_config: DataWriteConfig) -> None:
        """Default implementation of the set_write_config method.

        Provide a method to customize the write configuration
        used by the step's write_outputs method if desired.

        Parameters
        ----------

        data_config : DataWriteConfig
            A typed dictionary defining the data_format (i.e., parquet, csv, etc.) 
            and any particular data_format_args available in the pyarrow API.
        """
        self._data_write_config = data_config

    def get_parameters(self) -> dict:
        """Default implementation of the get_parameters method.

        Called by the pipeline's get_parameters method. Implement a custom
        method if needed, such as removing large items from the dictionary.

        Returns
        -------

        dict
            A dictionary copy of the object's configuration.
        """
        params = vars(self).copy()
        return params

    def write_data(
        self,
        write_path: str,
        data: Union[pa.Table, None],
        data_filename: Union[str, None] = None,
    ) -> None:
        """Default implementation of the write_data method.

        Reuse existing write patterns that exist in the data handlers.

        Parameters
        ----------

        write_path : str 
            A directory where a file will be written.
        data : Union[pa.Table, None]
            PyArrow Table of data.
        data_filename : Union[str, None]
            Overwrite default file name (no filetype extension).
        """
        # Add the filename with the appropriate filetype extension to the write path
        if not data_filename:
            data_filename = f"{self.__class__.__name__}Data"
        file_path = os.path.join(
            write_path, (f'{data_filename}.{self._data_write_config["data_format"]}')
        )
        # Call the data handler write_file function
        data_handlers.write_file(
            data=data,
            path=file_path,
            format=self._data_write_config["data_format"],
            **self._data_write_config["data_format_args"],
        )

    def write_artifacts(self, write_path: str, artifacts: Union[dict, None]):
        """Default implementation of the write_artifacts method.

        Since only some steps require this method, the default behavior is 'pass'.

        Implement a custom method to write the specific artifacts produced
        in the step.

        Parameters
        ----------

        write_path : str
            A directory where a file should be written.
        artifacts : Union[dict, None]
            A dictionary of artifacts from the step that should be written to a file
        """
        pass


class Pipeline:
    """Organize Steps into a pipeline to operate on data from a Dataset"""

    def __init__(
        self,
        dataset: Dataset = None,
        write_path: str = None,
        write_outputs: Literal["pipeline-outputs", "debug", False] = "pipeline-outputs",
        data_write_config: DataWriteConfig = None,
        data_filename: str = "bardi_processed_data",
    ):
        """Create a pipeline for organizing data pre-processing steps.

        Parameters
        ----------

        dataset : bardi.data.data_handlers.Dataset
            A bardi dataset object with data to pre-process.
        write_path : str
            Directory in the file system to write outputs to.
        write_outputs : Literal["pipeline-outputs", "debug", False]
            Configuration for which outputs to write.
                *   'pipeline-outputs' will write all artifacts and final data.
                *   'debug' will write all artifacts and data from each step.
                *   False will not write any files.
        data_write_config : DataWriteConfig
            Supply a custom write configuration specifying
            file types. Default will save data as parquet files.
        data_filename : str
            Supply a filename for the final output data.
        """
        # Reference a bardi dataset object that the pipeline will operate on
        self.dataset: Dataset = dataset

        # Pipeline configuration
        self.steps: List[Step] = []
        self.num_steps = 0
        self.write_outputs = write_outputs
        self.write_path = write_path
        self.data_filename = data_filename

        # File writing configuration
        if data_write_config:
            self.data_write_config = data_write_config
        else:
            self.data_write_config = {
                "data_format": "parquet",
                "data_format_args": {"compression": "snappy", "use_dictionary": False},
            }  # Default config, but can be overwritten
        if write_outputs:
            if not os.path.isdir(self.write_path):
                raise ValueError(
                    "Pipeline write_outputs was configured to "
                    "write, however a write_path was not "
                    "correctly specified. Please provide a "
                    "path to a directory to write files into."
                )

        # Tracking information produced during pipeline execution
        self.artifacts = {}
        self.performance = {}

    def add_step(self, step: Step) -> None:
        """Adds a step object to the list of pipeline execution steps.

        Also overwrites the step's data write configuration with a consistent
        pipeline data write configuration.

        Parameters
        ----------

        step : bardi.pipeline.Step
            A bardi Step object to be added to the list of pipeline execution steps.
        """
        if isinstance(step, Step):
            step.set_write_config(data_config=self.data_write_config)
            self.steps.append(step)
            self.num_steps += 1
        else:
            raise TypeError(
                "Only objects of type Step may be added to " "a bardi pipeline."
            )

    def run_pipeline(self) -> None:
        """Calls the run and write_outputs method for each respective step
        object added to the pipeline.
        """

        if isinstance(self.dataset, Dataset):
            self.processed_data = self.dataset.data
        else:
            raise TypeError(
                "Pipeline's Dataset must reference a bardi "
                "Dataset object. Please utilize bardi's data_"
                "handlers to create a bardi Dataset object with "
                "your data."
            )

        # For each step of the pipeline, call its run method
        pipeline_start_time = datetime.now()
        for step_position, step in enumerate(self.steps, start=1):
            # Call the run method and time the execution
            step_start_time = datetime.now()
            tracemalloc.start()
            results = step.run(data=self.processed_data, artifacts=self.artifacts)
            step_max_mem = tracemalloc.get_traced_memory()[1] / 1000000
            tracemalloc.stop()
            step_end_time = datetime.now()
            step_run_time = step_end_time - step_start_time

            # Record the performance
            if self.write_outputs == "debug":
                print(f"{str(type(step))} run time: {step_run_time}")
                print(f"{str(type(step))} max memory (MB): {step_max_mem}")
            self.performance[str(type(step))] = {
                "time": str(step_run_time),
                "memory (MB)": str(step_max_mem),
            }

            if isinstance(results, tuple):
                if isinstance(results[0], pa.Table):
                    # Set the pipeline's processed_data attribute to the newest result
                    self.processed_data = results[0]

                    # Write the step's data output to a file if pipeline is configured
                    # to do so
                    if self.write_outputs == "debug":
                        step.write_data(
                            write_path=self.write_path, data=self.processed_data
                        )
                    elif (
                        step_position == self.num_steps
                        and self.write_outputs == "pipeline-outputs"
                    ):
                        step.write_data(
                            write_path=self.write_path,
                            data=self.processed_data,
                            data_filename=self.data_filename,
                        )
                if isinstance(results[1], dict):
                    # If artifacts were returned by the step, add them to
                    # the pipeline's total set of artifacts
                    self.artifacts = {**self.artifacts, **results[1]}

                    # Write the step's artifacts to files if pipeline is configured
                    # to do so
                    if self.write_outputs:
                        step.write_artifacts(
                            write_path=self.write_path, artifacts=self.artifacts
                        )
            else:
                raise TypeError(
                    "Pipeline expected step to return a tuple of "
                    "PyArrow Table of data and a dictionary of "
                    "artifacts. If the step doesn't return one of "
                    "these, that position in the tuple can be "
                    "empty, but it still needs to return a tuple."
                )

        # Record the total pipeline performance
        pipeline_end_time = datetime.now()
        pipeline_run_time = pipeline_end_time - pipeline_start_time
        if self.write_outputs == "debug":
            print(f"Pipeline run time: {pipeline_run_time}")
        self.performance[str(type(self))] = str(pipeline_run_time)

    def get_parameters(self, condensed: bool = True) -> dict:
        """Returns the parameters of the pipeline's dataset and parameters of each step.

        Parameters
        ----------

        condensed : bool
            If True, the step configuration dictionary will exclude any
            attributes set to None or False.

        Returns
        -------

        dict
            Dictionary of parameters used to configure each pipeline step object.
        """

        pipeline_params = {"dataset": {}, "steps": {}, "performance": {}}

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
