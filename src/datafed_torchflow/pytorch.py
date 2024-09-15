import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import datetime
import traceback
from datafed.CommandLib import API
from os.path import basename
import zipfile
from datafed_torchflow.datafed import DataFed
from datafed.CommandLib import API
from datafed_torchflow.computer import get_system_info
import getpass
from datetime import datetime
from m3util.globus.globus import check_globus_file_access
from m3util.notebooks.checksum import calculate_notebook_checksum


class TorchLogger(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        DataFed_path,
        script_path=None,
        local_path="./",
        verbose=False,
    ):
        super(TorchLogger, self).__init__()
        self.__file__ = script_path
        self.model = model
        self.optimizer = optimizer
        self.DataFed_path = DataFed_path
        self.verbose = verbose
        self.local_path = local_path
        self.df_api = DataFed(self.DataFed_path)

        check_globus_file_access(self.df_api.endpointDefaultGet, self.local_path)

        self.save

    def getMetadata(self, **kwargs):
        # Get the current user and current time
        current_user = getpass.getuser()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Serialize model, optimizer, and get system info
        model = self.serialize_model()
        optimizer = self.serialize_pytorch_optimizer()
        computer_info = get_system_info()

        # Add current user and time to metadata
        metadata = (
            model
            | optimizer
            | computer_info
            | {"user": current_user, "timestamp": current_time}
            | kwargs
        )

        if self.__file__ is not None:
            script_checksum = calculate_notebook_checksum(self.__file__)
            file_info = {"script": {"path": self.__file__, "checksum": script_checksum}}
            metadata |= file_info

        return metadata

    def save(self, metadata, record_file_name, datafed=True, **kwargs):
        path = f"{self.local_path}/{record_file_name}"

        torch.save(self.model.state_dict(), path)

        if datafed:
            self.getMetadata(**kwargs)
            dc_resp = self.df_api.data_record_create(
                metadata, record_file_name.split(".")[0]
            )
            self.df_api.upload_file(dc_resp, path)
            
    def serialize_model(self):
        model_info = {}
        model_info['layers'] = {}
        
        top_level_count = 0  # Counter for the top-level layers
        num_top_level_layers = len(set([layer_name.split('.')[0] for layer_name, _ in self.model.named_modules() if layer_name != ""]))
        pad_length = len(str(num_top_level_layers))  # Padding length for top-level numbering

        for layer_name, layer in self.model.named_modules():
            # Skip the top layer which is the entire model itself
            if layer_name != "":
                # Split the layer name at the first '.'
                parts = layer_name.split('.', 1)
                top_level_name = parts[0]
                sub_name = parts[1] if len(parts) > 1 else None
                
                # Check if this is a new top-level layer (ignoring numbering)
                if top_level_name not in [key.split('-', 1)[1] for key in model_info['layers'].keys()]:
                    # Increment the top-level counter
                    top_level_count += 1
                    padded_count = str(top_level_count).zfill(pad_length)
                    
                    # Create a top-level entry with zero-padded numbering
                    model_info['layers'][f"{padded_count}-{top_level_name}"] = {}

                # Reference the top-level dictionary without the number prefix
                current_level_key = [key for key in model_info['layers'].keys() if key.endswith(f"-{top_level_name}")][0]
                current_level = model_info['layers'][current_level_key]
                
                # If there is a sub_name (nested), create nested entries
                if sub_name:
                    sub_parts = sub_name.split('.')
                    for sub in sub_parts[:-1]:
                        if sub not in current_level:
                            current_level[sub] = {}
                        current_level = current_level[sub]
                    layer_key = sub_parts[-1]
                else:
                    layer_key = top_level_name

                # Collect the layer information
                layer_descriptor = {
                    'type': layer.__class__.__name__,
                    'layer_name': layer_name,
                    'config': {}
                }

                # Automatically collect layer parameters
                for param, value in layer.__dict__.items():
                    # Filter out unnecessary attributes
                    if not param.startswith('_') and not callable(value):
                        layer_descriptor['config'][param] = value

                # Add the layer descriptor under the correct key
                current_level[layer_key] = layer_descriptor

        return model_info
    
    def serialize_pytorch_optimizer(self):
        state_dict = self.optimizer.state_dict()
        state_dict_serializable = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict_serializable[key] = value.tolist()
            # elif isinstance(value, dict):
            #     state_dict_serializable[key] = serialize_pytorch_optimizer(value)
            elif isinstance(value, list):
                state_dict_serializable[key] = [v.tolist() if isinstance(v, torch.Tensor) else v for v in value]
            else:
                state_dict_serializable[key] = value
        return state_dict_serializable['param_groups']
