import sys
import os


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime

sys.path.append(os.path.abspath("/home/jg3837/DataFed_TorchFlow/DataFed_TorchFlow/src"))


from datafed_torchflow.datafed import DataFed
from datafed_torchflow.computer import get_system_info
from datafed_torchflow.utils import (
    extract_instance_attributes,
    getNotebookMetadata,
    serialize_model,
    serialize_pytorch_optimizer,
)

import getpass
from datetime import datetime
from m3util.globus.globus import check_globus_file_access
from m3util.util.IO import find_files_recursive, make_folder

import json
from tqdm import tqdm
import logging
import numpy as np
import inspect
from datafed.CommandLib import API
import pathlib
import types
import ast
import traceback


# TODO: Make it so it does not upload a notebook on each reinstantiation. Compare the notebook checksum to determine whether the file has changed.
# TODO: Add data and dataloader derivative.


class TorchLogger:
    """
    TorchLogger is a class designed to log PyTorch model training details,
    including model architecture, optimizer state, and system information.
    It also integrates with the DataFed API for file and metadata management.

    Attributes:
        model_dict (dict): a dictionary containing the Pytorch model architecture to be logged,
            with the name of the block as the key and the block as the value.
            For example: {"vae":vae, "encoder: encoder, "decoder":decoder,"optimizer":optimizer}
        DataFed_path (str): The path to the DataFed configuration or API.
        script_path (str): Path to the script or notebook for checksum calculation.
        local_model_path (str): Local directory to store model files.
        input_data_shape (tuple): Shape of the input training data for the model.
        logging (bool): Whether to display logging output.

    """

    def __init__(
        self,
        model_dict,
        DataFed_path,
        script_path=None,
        local_model_path="/.",
        log_file_path="log.txt",
        input_data_shape=None,
        dataset_id=None,
        logging=False,
    ):
        """
        Initializes the TorchLogger class.

        Args:
            model_dict (dict): a dictionary containing the Pytorch model architecture to be logged,
                with the name of the block as the key and the block as the value.
                For example: {"vae":vae, "encoder: encoder, "decoder":decoder,"optimizer":optimizer}
            DataFed_path (str): Path to the DataFed configuration or API.
            script_path (str, optional): Path to the script or notebook. Default is None.
            local_model_path (str, optional): Local directory to store model files. Default is './'.
            log_file_path (str, optional): Local file to store a log of the code evaluation. Default is 'log.txt'
            input_data (numpy.ndarray, default=None): Input data for training the model.
            dataset_id (str, default=None): DataFed ID for the input dataset for the model
            logging (bool, optional): Flag for logging output. Default is False.
        """

        self.current_checkpoint_id = None
        self.notebook_record_id = None
        self.__file__ = script_path
        self.model_dict = model_dict
        self.optimizer = self.model_dict["optimizer"]
        self.DataFed_path = DataFed_path
        self.local_model_path = local_model_path
        self.log_file_path = log_file_path

        self.logging = logging
        self.input_data_shape = input_data_shape

        make_folder(self.local_model_path)

        self.df_api = DataFed(
            self.DataFed_path,
            self.local_model_path,
            log_file_path=self.log_file_path,
            logging=True,
        )
        self.dataset_id = dataset_id

        # Check if Globus has access to the local path
        check_globus_file_access(self.df_api.endpointDefaultGet, self.local_model_path)

        # Save the notebook to DataFed
        self.save_notebook()

    def reset(self):
        self.current_checkpoint_id = None
    
    @property
    def optimizer(self):
        """
        Returns the optimizer used for training.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Sets the optimizer used for training.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to be set.
        """
        self._optimizer = optimizer

    def getMetadata(self, local_vars=None, model_hyperparameters=None, **kwargs):
        """
        Gathers metadata including the serialized model, optimizer, system info, and user details.

        Args:
            local_vars (list): a list containing the local variables for the model training code, from list(locals().items()). Used to determine the metadata
            **kwargs: Additional key-value pairs to be added to the metadata.

        Returns:
            dict: A dictionary containing the metadata including model, optimizer,
                  system information, user, timestamp, and optional script checksum.
        """

        # get the model architecture names
        model_architecture_names = list(self.model_dict.keys())

        # get the user information and timestamp
        current_user, current_time = self.getUserClock()

        # get the computer information
        computer_info = get_system_info()

        # combine metadata with user and timestamp

        # define the empty metadata dictionary to set the structure
        DataFed_record_metadata = {
            "Model Parameters": {"Model Hyperparameters": {}, "Model Architecture": {}},
            "System Information": {},
        }

        # loop through the local variables to add to the metadata dictionary
        for key, value in local_vars:
            # exclude modules and other undesired local variables. Use casefold string matching for flexibility
            if (
                not key.startswith("_")
                and key.casefold()
                not in [
                    "checkpoint",
                    "self",
                    "local_vars",
                    "model_dict",
                    "key",
                    "value",
                ]
                and "datafed" not in key.casefold()
                and "globus" not in key.casefold()
                and "data".casefold() not in str(type(value)).casefold()
                and "dataloader".casefold() not in str(type(value)).casefold()
                and not (callable(value) and key not in model_architecture_names)
                and type(value)
                not in [
                    type,
                    types.ModuleType,
                    types.FunctionType,
                    API,
                    DataLoader,
                    types.NoneType,
                    types.MethodType,
                ]
            ):
                # put the model architecture into the Model Architecture sub-dictionary
                if key in model_architecture_names:
                    # serialize the optimizer
                    if key.casefold() in [
                        "optimizer",
                        "optim",
                    ]:  # accept "optimizer" or "optim" for flexibility
                        DataFed_record_metadata["Model Parameters"][
                            "Model Architecture"
                        ][key] = serialize_pytorch_optimizer(value)
                    # serialize the model architecture blocks (encoder, decoder, etc. )
                    else:
                        DataFed_record_metadata["Model Parameters"][
                            "Model Architecture"
                        ][key] = serialize_model(value)
                        DataFed_record_metadata["Model Parameters"][
                            "Model Architecture"
                        ][key].update(extract_instance_attributes(obj=value))
                # extract lists if they are not too long (arbitrarily chose to be less than 1000 characters)
                elif isinstance(value, list):
                    # ignore long lists
                    if sum(len(str(s)) for s in value) < 1000:
                        # extract the value for 1 item lists
                        if len(value) == 1:
                            DataFed_record_metadata["Model Parameters"][key] = value[0]
                        else:
                            # if the list has many (but not too many values) extract the whole list
                            DataFed_record_metadata["Model Parameters"][key] = value
                    else:
                        warning_message = (
                            f'List in key "{key}" is too long to be extracted'
                        )
                        Warning(warning_message)
                # put the model hyperparameters in the Model Hyperparameters sub-dictionary (the hyperparameters might be 1-value torch tensors or just floats)
                elif key in model_hyperparameters.keys():
                    if type(value) in [np.ndarray, torch.Tensor]:
                        DataFed_record_metadata["Model Parameters"][
                            "Model Hyperparameters"
                        ][key] = value.tolist()
                    else:
                        DataFed_record_metadata["Model Parameters"][
                            "Model Hyperparameters"
                        ][key] = value

                # convert numpy arrays and torch tensors that are small enough (arbitrarily chosen to be smaller than the input data dimensions)
                # into lists so they can be serialized into JSON
                elif type(value) in [np.ndarray, torch.Tensor]:
                    if value.shape < self.input_data_shape:
                        # put other lists into the Model Parameters dictionary
                        DataFed_record_metadata["Model Parameters"][key] = (
                            value.tolist()
                        )
                # convert PosixPaths and pytorch devices into strings so they can be serialized into JSON
                elif type(value) in [pathlib.PosixPath, torch.device]:
                    DataFed_record_metadata["Model Parameters"][key] = str(value)
                # convert class instances into dictionaries of their attributes so they can be serialized into JSON
                elif hasattr(value, "__dict__"):
                    if len(extract_instance_attributes(obj=value)) > 0:
                        DataFed_record_metadata["Model Parameters"][key] = (
                            extract_instance_attributes(obj=value)
                        )

                elif isinstance(value, dict):
                    if "_" not in str(type(value[list(value.keys())[0]])):
                        try:
                            json.dumps(value)
                            DataFed_record_metadata["Model Parameters"][key] = value
                        except (TypeError, ValueError, json.JSONDecodeError):
                            DataFed_record_metadata["Model Parameters"][key] = str(
                                value
                            )

                # all other cases, everything should be serializable (string, float, etc.)
                else:
                    # everything should be JSON serializable at this point, but try to convert to string and then skip if not

                    try:
                        json.dumps(value)
                        DataFed_record_metadata["Model Parameters"][key] = value
                    except (TypeError, ValueError, json.JSONDecodeError):
                        try:
                            DataFed_record_metadata["Model Parameters"][key] = str(
                                value
                            )

                        except (TypeError, ValueError, json.JSONDecodeError):
                            if self.logging:
                                tb = traceback.format_exc()
                                with open(self.log_file_path, "a") as f:
                                    timestamp = (
                                        datetime.now()
                                        .astimezone()
                                        .strftime("%Y-%m-%d %H:%M:%S")
                                    )

                                    f.write(
                                        f"\n {timestamp} - Could not convert {key} to JSON. {key} has type {type(key)}"
                                    )
                                    f.write(
                                        f"the corresponding value has type {type(value)} and value \n {value}"
                                    )
                                    f.write(f"Python error message {tb}")
                                    f.write("skipping this variable.")

                #  except:
                #      pass
        # add the notebook checksum and file path to the Model Parameters dictionary
        DataFed_record_metadata["Model Parameters"].update(
            getNotebookMetadata(self.__file__)
        )
        # add the user and timestamp to the Model Parameters dictionary
        DataFed_record_metadata["Model Parameters"].update(
            {"user": current_user, "timestamp": current_time}
        )
        # add the computer information to the System Information section
        DataFed_record_metadata["System Information"] = computer_info

        # return the metadata
        return DataFed_record_metadata

    def getModelArchitectureStateDict(self):
        """
        generates a dictionary where the key is the model architecture block
        and the value is the corresponding state dictionary to go in the saved checkpoint,
        for example

        Returns:
            dict: A dictionary containing the model architecture state dictionaries

        """
        model_architecture = {}

        for block in self.model_dict.keys():
            model_architecture[block] = self.model_dict[block].state_dict()

        return model_architecture

    def getUserClock(self):
        """
        Gathers system information including CPU, memory, and GPU details.

        Returns:
            dict: A dictionary containing system information.
        """
        # Get the current user
        current_user = getpass.getuser()

        # Get the current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return current_user, current_time

    def save_notebook(self):
        """
        Saves the Jupyter notebook that runs the code training the model
        """
        # don't upload the notebook to DataFed if it is already there. NOTE: The below method to check is a temporary solution and will be replaced with a comparison of the checksums
        # first, check if the notebook filename is actually its DataFed ID, in which case it already exists in DataFed
        if self.__file__.startswith("d/"):
            self.notebook_record_id = self.__file__

        # if the notebook filename is not a DataFed ID, check if a notebook of the same name exists at the DataFed file path
        elif self.__file__ is not None:
            try:
                # this will fail if it doesn't find a match, meaning that the notebook does not already exists on DataFed
                self.notebook_record_id = (
                    self.df_api.get_notebook_DataFed_ID_from_path_and_title(
                        self.__file__
                    )
                )

            except Exception as e:
                # the notebook is not already in DataFed, so upload it
                # output to user
                if self.logging:
                    with open(self.log_file_path, "a") as f:
                        timestamp = (
                            datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        f.write(
                            f"\n {timestamp} - Uploading notebook {self.__file__} to DataFed..."
                        )

                notebook_metadata = getNotebookMetadata(self.__file__)

                current_user, current_time = self.getUserClock()

                notebook_metadata = notebook_metadata | {
                    "user": current_user,
                    "timestamp": current_time,
                }

                self.notebook_record_resp = self.df_api.data_record_create(
                    metadata=notebook_metadata,
                    record_title=self.__file__.split("/")[-1],  # .split(".")[0],
                    deps=self.df_api.addDerivedFrom(self.dataset_id),
                )

                self.df_api.upload_file(
                    self.notebook_record_resp[0].data[0].id, self.__file__
                )

                self.notebook_record_id = self.notebook_record_resp[0].data[0].id

    def save(
        self,
        record_file_name,
        datafed=True,
        local_file_path=None,
        local_vars=None,
        model_hyperparameters=None,
        **kwargs,
    ):
        """


        Saves the model's state dictionary locally unless one has already been saved
        and optionally uploads it to DataFed along with the model's metadata.
        If you want to upload multiple files to the same DataFed data record you can zip them
        together and pass in the local path to the zip file as "local_file_path".

        Args:
            record_file_name (str): The name of the file to save the model locally.
            datafed (bool, optional): If True, the record is uploaded to DataFed. Default is True.
            local_file_path (str or Path.PosixPath, optional): The local file path to the directory to save the weights
                    or to the presaved file to upload to DataFed.
            local_vars (list): a list containing the local variables for the model training code, from list(locals().items()). Used to determine the metadata
            model_hyperparameters (dict): a dictionary where the keys are the model hyperparameters names and the values are the model hyperparameter names. Used in the saved checkpoint.
            **kwargs: Additional metadata or attributes to include in the record.
        """

        # include the model architecture state dictionary and model hyperparameters in the checkpoint
        if not str(local_file_path).endswith(".zip") and not os.path.exists(
            str(local_file_path)
        ):
            checkpoint = self.getModelArchitectureStateDict() | model_hyperparameters

            # Save the model state dict locally
            torch.save(checkpoint, local_file_path)

        if datafed:
            # Safely retrieve values and replace with None if undefined or not present
            notebook_record_id = (
                self.notebook_record_id
                if self.notebook_record_id and len(self.notebook_record_id) > 0
                else None
            )

            # Saves the record id to the object
            self.notebook_record_id = notebook_record_id

            current_checkpoint_id = (
                self.current_checkpoint_id
                if self.current_checkpoint_id is not None
                else None
            )

            # Create a list of IDs, excluding any that are None
            ids_to_add = [
                id
                for id in [notebook_record_id, current_checkpoint_id, self.dataset_id]
                if id is not None
            ]

            # Call the API method with the valid IDs (if any)
            if ids_to_add:
                deps = self.df_api.addDerivedFrom(ids_to_add)
            else:
                deps = None  # If no valid IDs are present, set deps to None

            # Generate metadata and create a data record in DataFed
            metadata = self.getMetadata(
                local_vars=local_vars,
                model_hyperparameters=model_hyperparameters,
                **kwargs,
            )

            dc_resp = self.df_api.data_record_create(
                metadata,
                record_title=str(record_file_name),
                local_model_path=self.local_model_path,
                # weights_file_path = weights_file_path,
                # embedding_file_path = embedding_file_path,
                # reconstruction_file_path = reconstruction_file_path,
                deps=deps,
            )
            # Upload the saved model to DataFed
            self.df_api.upload_file(dc_resp[0].data[0].id, str(local_file_path))

            self.current_checkpoint_id = dc_resp[0].data[0].id

    def get_return_variables(func):
        # Get the source code of the function
        source = inspect.getsource(func)

        # Parse the source code into an AST
        tree = ast.parse(source)

        # Navigate to the function definition in the AST
        function_node = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef)
        )

        # Extract the return statement
        return_vars = []
        for node in ast.walk(function_node):
            if isinstance(node, ast.Return):
                # Check if the return value is a tuple or a single value
                if isinstance(node.value, ast.Tuple):
                    return_vars = [
                        elt.id for elt in node.value.elts if isinstance(elt, ast.Name)
                    ]
                elif isinstance(node.value, ast.Name):
                    pass
                break


class InferenceEvaluation:
    def __init__(
        self,
        dataframe,
        dataset,
        df_api,
        root_directory=None,
        save_directory="./tmp/",
        skip=None,
        **Kwargs,
    ):
        self.df = dataframe
        self.dataset_id = dataset
        self.root_directory = root_directory
        self.save_directory = save_directory
        self.df_api = df_api
        self.skip = skip

        self.model = self.build_model(**Kwargs)

        # Create a logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.WARNING)

    def file_not_found(self, filename, row):
        self.logger.warning(
            "{filename} was not found from DataFed using record id {row.id}"
        )

        print(
            f"Attempting to download {filename} from DataFed using record id {row.id}"
        )

        ds_rep = self.df_api.dataGet(row.id, self.save_directory, wait=True)

        if ds_rep[0].task[0].status == 3:
            # Download was successful
            self.logger.info(f"{filename} was downloaded successfully")

            # finds the downloaded file recursively
            file_path = find_files_recursive(self.root_directory, filename)

            return file_path

        else:
            # Download was not successful
            self.logger.error(f"{filename} could not be downloaded")

            # returns a None object
            return None

        return ds_rep

    def _getFileName(self, row):
        return self.df_api.getFileName(row.id)

    @staticmethod
    def get_first_entry_if_list(data):
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # Return the first entry if it's a non-empty list
        else:
            return data

    def run_inference(self, row):
        # retrive the filename from the API datarecords
        filename = self._getFileName(row)

        # checks if the file can be found in the root directory
        file_path = find_files_recursive(self.root_directory, filename)

        if len(file_path) == 0:
            # if the file is not found, attempt to download it from DataFed
            file_path = self.file_not_found(filename, row)

            file_path = self.get_first_entry_if_list(file_path)

            if file_path is None:
                self.logger.info(
                    f"{filename} could not be downloaded, skipping inference."
                )

                print(f"{filename} could not be downloaded, skipping inference.")

                return None

        # load the model
        self.model.load(file_path[0])

        return self.evaluate(row, file_path)

    def build_model(self):
        """
        Builds and returns the model to be used for inference.

        This method should be implemented by the child class to define the specific model architecture
        and any necessary configurations.

        Returns:
            torch.nn.Module: The model object to be used for inference.
        """
        raise NotImplementedError(
            "Child class must implement this method. This method should return a model object."
        )

    def evaluate(self, row, file_path):
        """
        Evaluates the model on the given data. This method should be implemented by the child class.
        The parent class does not implement this method.

        Args:
            row (pd.Series): A row from the dataframe containing metadata and other information.
            file_path (str): The path to the file to be used for evaluation.

        Returns:
            dict: The evaluation results as a dictionary.
        """
        raise NotImplementedError(
            "Child class must implement this method. This method should return evaluation results as a dictionary."
        )

    def run(self):
        for i, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # set to restart and skip
            if self.skip is not None and i <= self.skip:
                continue

            # runs the inference
            msg = self.run_inference(row)

            # if file cannot be found, skip inference
            if msg is None:
                continue

            # updates the metadata of the record
            self.df_api.dataUpdate(row.id, metadata=json.dumps(msg))

            # logs the success of the inference
            self.logger.info(f"Inference for {i} record {row.id} was successful")


class TorchViewer(nn.Module):
    def __init__(self, DataFed_path):
        self.DataFed_path = DataFed_path
        self.df_api = DataFed(self.DataFed_path)

    def getModelCheckpoints(
        self,
        exclude_metadata="computing",
        excluded_keys="script",
        non_unique=["id", "timestamp", "total_time"],
        format="pandas",
    ):
        """
        Retrieves the metadata record for a specified record ID.

        Args:
            record_id (str): The ID of the record to retrieve.
            exclude_metadata (str, list, or None, optional): Metadata fields to exclude from the extraction record.
            excluded_keys (str, list, or None, optional): Keys if the metadata record contains to exclude.
            non_unique (str, list, or None, optional): Keys which are expected to be unique independent of record uniqueness - these are not considered when finding unique records.
            format (str, optional): The format to return the metadata in. Defaults to "pandas".

        Returns:
            dict: The metadata record.
        """

        return self.df_api.get_metadata(
            exclude_metadata=exclude_metadata,
            excluded_keys=excluded_keys,
            non_unique=non_unique,
            format=format,
        )
