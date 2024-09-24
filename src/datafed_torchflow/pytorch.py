import torch
import torch.nn as nn
import datetime
from datafed_torchflow.datafed import DataFed
from datafed_torchflow.computer import get_system_info
import getpass
from datetime import datetime
from m3util.globus.globus import check_globus_file_access
from m3util.notebooks.checksum import calculate_notebook_checksum
from m3util.util.IO import find_files_recursive
import json
from tqdm import tqdm
import logging
import numpy as np

# TODO: Make it so it does not upload a notebook on each reinstantiation. Checksum just the file.
# TODO: Add data and dataloader derivative.

# TODO: do I need this? 
class TorchLogger:
    """
    TorchLogger is a class designed to log PyTorch model training details,
    including model architecture, optimizer state, and system information.
    It also integrates with the DataFed API for file and metadata management.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be logged.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        DataFed_path (str): The path to the DataFed configuration or API.
        script_path (str): Path to the script or notebook for checksum calculation.
        local_path (str): Local directory to store model files.
        verbose (bool): Whether to display verbose output.
        df_api (DataFed): Instance of the DataFed API client for managing data records.
    """

    def __init__(
        self,
        model,
        DataFed_path,
        optimizer=None,
        script_path=None,
        local_path="./",
        verbose=False,
        notebook_metadata=None,
        dataset_id=None,
    ):
        """
        Initializes the TorchLogger class.

        Args:
            model (torch.nn.Module): The PyTorch model to log.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            DataFed_path (str): Path to the DataFed configuration or API.
            script_path (str, optional): Path to the script or notebook. Default is None.
            local_path (str, optional): Local directory to store model files. Default is './'.
            verbose (bool, optional): Flag for verbose output. Default is False.
        """
        self.current_checkpoint_id = None
        self.notebook_record_id = None
        self.notebook_metadata = notebook_metadata
        self.__file__ = script_path
        self.model = model
        self.optimizer = optimizer
        self.DataFed_path = DataFed_path
        self.verbose = verbose
        self.local_path = local_path
        self.df_api = DataFed(self.DataFed_path)
        self.dataset_id = dataset_id

        # Check if Globus has access to the local path
        check_globus_file_access(self.df_api.endpointDefaultGet, self.local_path)

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

    def getMetadata(self, **kwargs):
        """
        Gathers metadata including the serialized model, optimizer, system info, and user details.

        Args:
            **kwargs: Additional key-value pairs to be added to the metadata.

        Returns:
            dict: A dictionary containing the metadata including model, optimizer,
                  system information, user, timestamp, and optional script checksum.
        """

        current_user, current_time = self.getUserClock()

        # Serialize model, optimizer, and get system info
        model = self.serialize_model()
        optimizer = self.serialize_pytorch_optimizer()
        computer_info = get_system_info()

        # Combine metadata and add user and timestamp
        metadata = (
            model
            | {"optimizer": optimizer}
            | computer_info
            | {"user": current_user, "timestamp": current_time}
            | kwargs
        )

        # Check if the script path is provided and does not start with "d/"
        if self.__file__ is not None and not self.__file__.startswith("d/"):
            # If notebook metadata is not already set, calculate and set it
            if self.notebook_metadata is None:
                self.notebook_metadata = self.getNotebookMetadata()

        if self.notebook_metadata is not None:
            metadata |= self.notebook_metadata
            self.notebook_id = self.__file__

        return metadata

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

    def getNotebookMetadata(self):
        """
        Calculates the checksum of the script or notebook file and includes it in the metadata.

        Returns:
            dict: A dictionary containing the path and checksum of the script or notebook file.
        """

        # If the script path is provided, calculate and include its checksum
        if self.__file__ is not None:
            script_checksum = calculate_notebook_checksum(self.__file__)
            file_info = {"script": {"path": self.__file__, "checksum": script_checksum}}
            return file_info

    def save_notebook(self):
        if self.__file__.startswith("d/"):
            self.notebook_record_id = self.__file__
            
        elif self.__file__ is not None:
            
            
            try: 
                self.notebook_record_id = self.df_api.get_notebook_DataFed_ID_from_path_and_title(self.__file__)
            
            except:
           
                # output to user
                if self.verbose:
                    print(f"Uploading notebook {self.__file__} to DataFed...")

                notebook_metadata = self.getNotebookMetadata()

                current_user, current_time = self.getUserClock()

                notebook_metadata = notebook_metadata | {
                    "user": current_user,
                    "timestamp": current_time,
                }

                self.notebook_record_resp = self.df_api.data_record_create(
                    notebook_metadata,
                    self.__file__.split("/")[-1], #.split(".")[0],
                    deps=self.df_api.addDerivedFrom(self.dataset_id),
                )

                self.df_api.upload_file(self.notebook_record_resp, self.__file__)

                self.notebook_record_id = self.notebook_record_resp[0].data[0].id

    def save(self, record_file_name, training_loss=None, datafed=True, **kwargs):
        """
        Saves the model's state dictionary locally and optionally uploads it to DataFed.

        Args:
            metadata (dict): Metadata to be associated with the model record.
            record_file_name (str): The name of the file to save the model locally.
            datafed (bool, optional): If True, the record is uploaded to DataFed. Default is True.
            training_loss (str, optional): The final training loss to include in the metadata.
            **kwargs: Additional metadata or attributes to include in the record.
        """
        path = f"{self.local_path}/{record_file_name}"

        # Save the model state dict locally
        torch.save(self.model.state_dict(), path)

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
            metadata = self.getMetadata(**kwargs)
            dc_resp = self.df_api.data_record_create(
                metadata,
                str(record_file_name),
                deps=deps,
            )
            # Upload the saved model to DataFed
            self.df_api.upload_file(dc_resp, path)

            self.current_checkpoint_id = dc_resp[0].data[0].id
            
            if training_loss is not None:
                dc_resp = self.df_api.data_record_create(
                    metadata,
                    f"Training_loss_{record_file_name}",
                    deps=self.df_api.addDerivedFrom(self.current_checkpoint_id),
                )
                
                # Upload the saved model to DataFed
                self.df_api.upload_file(dc_resp, training_loss)

    def serialize_model(self):
        """
        Serializes the model architecture into a dictionary format with detailed layer information.

        Returns:
            dict: A dictionary containing the model's architecture with layer types,
                  names, and configurations.
        """
        model_info = {}
        model_info["layers"] = {}

        top_level_count = 0  # Counter for the top-level layers
        num_top_level_layers = len(
            set(
                [
                    layer_name.split(".")[0]
                    for layer_name, _ in self.model.named_modules()
                    if layer_name != ""
                ]
            )
        )
        pad_length = len(
            str(num_top_level_layers)
        )  # Padding length for top-level numbering

        for layer_name, layer in self.model.named_modules():
            # Skip the top layer which is the entire model itself
            if layer_name != "":
                # Split the layer name at the first '.'
                parts = layer_name.split(".", 1)
                top_level_name = parts[0]
                sub_name = parts[1] if len(parts) > 1 else None

                # Check if this is a new top-level layer (ignoring numbering)
                if top_level_name not in [
                    key.split("-", 1)[1] for key in model_info["layers"].keys()
                ]:
                    # Increment the top-level counter
                    top_level_count += 1
                    padded_count = str(top_level_count).zfill(pad_length)

                    # Create a top-level entry with zero-padded numbering
                    model_info["layers"][f"{padded_count}-{top_level_name}"] = {}

                # Reference the top-level dictionary without the number prefix
                current_level_key = [
                    key
                    for key in model_info["layers"].keys()
                    if key.endswith(f"-{top_level_name}")
                ][0]
                current_level = model_info["layers"][current_level_key]

                # If there is a sub_name (nested), create nested entries
                if sub_name:
                    sub_parts = sub_name.split(".")
                    for sub in sub_parts[:-1]:
                        if sub not in current_level:
                            current_level[sub] = {}
                        current_level = current_level[sub]
                    layer_key = sub_parts[-1]
                else:
                    layer_key = top_level_name

                # Collect the layer information
                layer_descriptor = {
                    "type": layer.__class__.__name__,
                    "layer_name": layer_name,
                    "config": {},
                }

                # Automatically collect layer parameters
                for param, value in layer.__dict__.items():
                    # Filter out unnecessary attributes
                    if not param.startswith("_") and not callable(value):
                        layer_descriptor["config"][param] = value

                # Add the layer descriptor under the correct key
                current_level[layer_key] = layer_descriptor

        return model_info

    def serialize_pytorch_optimizer(self):
        """
        Serializes the optimizer's state dictionary, converting tensors to lists for JSON compatibility.

        Returns:
            dict: A dictionary containing the optimizer's serialized parameters.
        """
        state_dict = self.optimizer.state_dict()
        state_dict_serializable = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict_serializable[key] = value.tolist()
            elif isinstance(value, list):
                # Convert tensors within lists to lists
                state_dict_serializable[key] = [
                    v.tolist() if isinstance(v, torch.Tensor) else v for v in value
                ]
            else:
                state_dict_serializable[key] = value
        return state_dict_serializable["param_groups"][0]


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
        self.logger.warning("{filename} was not found")

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

        print(filename)

        # checks if the file can be found in the root directory
        file_path = find_files_recursive(self.root_directory, filename)

        if len(file_path) == 0:
            # if the file is not found, attempt to download it from DataFed
            file_path = self.file_not_found(filename, row)

            file_path = self.get_first_entry_if_list(file_path)

            if file_path is None:
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
