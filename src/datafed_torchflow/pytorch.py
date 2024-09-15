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
