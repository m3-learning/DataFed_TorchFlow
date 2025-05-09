import json
import os
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
from datafed.CommandLib import API
from m3util.globus.globus import check_globus_endpoint
from tqdm import tqdm

from datafed_torchflow.JSON import UniversalEncoder


class DataFed(API):
    """
    A class to interact with DataFed API.

    Inherits from:
        API: The base class for interacting with the DataFed API.

    Attributes:
        datafed_path (str): DataFed path to store model script and checkpoints.
        local_model_path (str): Local directory to store model files.
        log_file_path (str): Local file to store a log of the code evaluation.
        logging (bool): Flag to enable logging.
        project_id (str): The ID of the project.
        dataset_id (str): The ID of the dataset.
    """

    def __init__(
        self,
        datafed_path: str,
        local_model_path="./Trained Models",
        log_file_path="log.txt",
        dataset_id_or_path: Optional[str] = None,
        download_kwargs={"wait": True, "orig_fname": True},
        upload_kwargs={"wait": True},
        logging=False,
    ):
        """
        Initializes the DataFed instance.

        Args:
            datafed_collection (str): The current DataFed collection could be provided as either a collection ID or a directory path.
            local_model_path (str): Local directory to store model files.
            log_file_path (str, optional): Local file to store a log of the code evaluation. Default is 'log.txt'
            logging (bool, optional): Flag to enable logging. Defaults to False.

        Raises:
            Exception: If the user is not authenticated with DataFed.
        """
        super().__init__()
        self.datafed_path = datafed_path

        self.local_model_path = local_model_path

        # sets the kwargs for downloads
        self.download_kwargs = download_kwargs

        # sets the kwargs for uploads
        self.upload_kwargs = upload_kwargs

        self.logging = logging
        self.log_file_path = log_file_path

        # checks if the user is autenticated with DataFed
        # and the Globus endpoint is set
        self.check_if_logged_in()
        self.check_if_endpoint_set()

        # gets the collection and project ID
        self.identify_collection_id()

        # Set the dataset ID or path
        self.dataset_id_or_path = dataset_id_or_path

        # Set the data path
        # self.data_path = data_path

    def upload_dataset_to_DataFed(self):
        """
        Checks whether the dataset record already exists on DataFed and uploads it to a collection called ``dataset" (which it will create if necessary)
        whose parent collection is self.collection_id (where the checkpoints are stored). Works with any number of dataset files, specified by
        torchlogger.dataset_id_or_path in the instantiation of the torchlogger. The dataset files can be specified as either their file names or
        DataFed IDs (specified as a string for a single dataset file and a list of strings for multiple dataset files)

        Args:
            None (self)

        Returns:
            The DataFed record ID for the dataset files, as a string for a single dataset file and a list of strings for multiple dataset files.
        """
        if self.dataset_id_or_path is not None:
            ls_resp = self.collectionItemsList(self.collection_id)

            if isinstance(
                self.dataset_id_or_path, list
            ):  # to specify multiple dataset files
                dataset_ids = []
                for dataset in self.dataset_id_or_path:
                    if dataset.startswith("d/"):
                        dataset_ids.append(dataset)
                    else:
                        try:
                            path_id = (
                                ls_resp[0]
                                .item[
                                    np.where(
                                        [
                                            record.title.casefold()
                                            == "Dataset".casefold()
                                            for record in ls_resp[0].item
                                        ]
                                    )[0].item()
                                ]
                                .id
                            )
                            # update record (dependencies have been added)
                            record_id = (
                                self.get_notebook_DataFed_ID_from_path_and_title(
                                    dataset, path_id=path_id
                                )
                            )

                            self.data_record_update(
                                record_id=record_id,
                                record_title=dataset,
                                parent_collection=path_id,
                            )

                            dataset_ids.append(record_id)

                            # dataset_ids.append(self.get_notebook_DataFed_ID_from_path_and_title(dataset, path_id=path_id))

                        except:
                            if "Dataset" not in self.getCollList(self.collection_id)[0]:
                                coll_resp = self.collectionCreate(
                                    "Dataset", parent_id=self.collection_id
                                )
                                # try:
                                #     # if get_notebook_DataFed_ID_from_path_and_title fails, the record doesn't already exist and should be created.
                                #     # Otherwise, update it. This should maybe be replaced with a checksum comparison eventually,
                                #     # although it probably won't change like the notebooks do.
                                #     record_id = self.get_notebook_DataFed_ID_from_path_and_title(dataset, coll_resp[0].coll[0].id )
                                #     self.data_record_update(record_id=record_id, record_title= dataset, parent_collection = coll_resp[0].coll[0].id)
                                # except:
                                dataset_id_resp = self.data_record_create(
                                    record_title=dataset,
                                    parent_collection=coll_resp[0].coll[0].id,
                                )

                            else:
                                dataset_id_resp = self.data_record_create(
                                    record_title=dataset,
                                    parent_collection=self.get_notebook_DataFed_ID_from_path_and_title(
                                        "Dataset"
                                    ),
                                )
                            dataset_id = dataset_id_resp[0].data[0].id
                            self.upload_file(dataset_id, dataset)
                            dataset_ids.append(dataset_id)

            else:  # type(dataset_id) is str, so only 1 dataset file specified
                if self.dataset_id_or_path.startswith("d/"):
                    dataset_ids = self.dataset_id_or_path
                else:
                    dataset = self.dataset_id_or_path  # Is this right?

                    try:
                        path_id = (
                            ls_resp[0]
                            .item[
                                np.where(
                                    [
                                        record.title.casefold() == "Dataset".casefold()
                                        for record in ls_resp[0].item
                                    ]
                                )[0].item()
                            ]
                            .id
                        )

                        # update record (dependencies have been added)
                        record_id = self.get_notebook_DataFed_ID_from_path_and_title(
                            dataset, path_id=path_id
                        )

                        self.data_record_update(
                            record_id=record_id,
                            record_title=dataset,
                            parent_collection=path_id,
                        )

                        dataset_ids = record_id

                    except:
                        if "Dataset" not in self.getCollList(self.collection_id)[0]:
                            coll_resp = self.collectionCreate(
                                "Dataset", parent_id=self.collection_id
                            )

                            dataset_id_resp = self.data_record_create(
                                record_title=dataset,
                                parent_collection=coll_resp[0].coll[0].id,
                            )

                        else:
                            dataset_id_resp = self.data_record_create(
                                record_title=dataset,
                                parent_collection=self.get_notebook_DataFed_ID_from_path_and_title(
                                    "Dataset"
                                ),
                            )
                        dataset_id = dataset_id_resp[0].data[0].id
                        self.upload_file(dataset_id, dataset)
                        dataset_ids = dataset_id
                    # try:
                    #     ls_resp = self.collectionItemsList(self.collection_id, count=100000)
                    #     path_id = ls_resp[0].item[np.where([record.title.casefold() == "Dataset".casefold() for record in ls_resp[0].item])[0].item()].id
                    #     dataset_ids = self.get_notebook_DataFed_ID_from_path_and_title(self.dataset_id_or_path, path_id = path_id)
                    # except:
                    #     dataset_id_resp = self.data_record_create(record_title = dataset)
                    #     dataset_ids = dataset_id_resp[0].data[0].id
                    #     self.upload_file(dataset_id, dataset)

            self.dataset_id = dataset_ids

        else:
            self.dataset_id = None

        return self.dataset_id

        # self.getData()

        # self.original_file_path = self.file_path

    def getCollectionProjectID(self):
        """
        Retrieves the project ID associated with a specific collection.

        This method fetches the parent collection of the given collection ID and extracts the project ID from it.

        Returns:
            str: The project ID associated with the specified collection.
        """

        # Fetch the parent collection of the specified collection ID
        parent_collection = self.collectionGetParents(self.collection_id)[0]

        # Extract and return the project ID from the parent collection path
        return parent_collection.path[0].item[0].id

    def identify_collection_id(self):
        # if a collection ID is provided, set the collection ID to the provided collection ID
        if self.datafed_path.startswith("c/"):
            self.collection_id = self.datafed_path
            self.project_id = self.getCollectionProjectID()

        # if provided as a path to a collection, set the collection ID to the collection ID of the last subfolder
        else:
            try:
                # Checks if the datafed_collection is a valid path.
                self.check_string_for_dot_or_slash(self.datafed_path)

                # Checks if user is saving in the root collection.
                if self._parse_datafed_collection[0] == self.user_id:
                    self.project_id = self.user_id
                else:
                    # Gets all the projects in DataFed.
                    items, response = self.get_projects

                    # Checks if the project exists in DataFed.
                    self.project_id = self.find_id_by_title(
                        items, self._parse_datafed_collection[0]
                    )

                    self.create_subfolder_if_not_exists()

            except ValueError:
                raise ValueError("Invalid DataFed collection path provided.")

    def check_if_logged_in(self):
        """
        Checks if the user is authenticated with DataFed.

        Raises:
            Exception: If the user is not authenticated.
        """
        if self.getAuthUser():
            if self.logging:
                with open(self.log_file_path, "a") as f:
                    timestamp = (
                        datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    f.write(
                        f"\n {timestamp} - Success! You have been authenticated into DataFed as: {self.getAuthUser()}."
                    )

        else:
            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

                f.write(
                    f"\n {timestamp} - You have not authenticated into DataFed Client."
                )
                f.write(
                    "Please follow instructions in the 'Basic Configuration' section in the link below to authenticate yourself:"
                )
                f.write(
                    "https://ornl.github.io/DataFed/user/client/install.html#basic-configuration"
                )

            raise Exception(
                "You have not authenticated into DataFed Client. Please follow instructions in the 'Basic Configuration' section in the link below to authenticate yourself: https://ornl.github.io/DataFed/user/client/install.html#basic-configuration"
            )

    def check_if_endpoint_set(self):
        """
        Checks if the Globus endpoint is set up.

        Raises:
            Exception: If the Globus endpoint is not set up.
        """
        if self.endpointDefaultGet():
            if self.logging:
                with open(self.log_file_path, "a") as f:
                    timestamp = (
                        datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    f.write(
                        f"\n {timestamp} - Success! You have set up the Globus endpoint {self.endpointDefaultGet()}."
                    )
        else:
            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

                f.write(
                    f"\n {timestamp} - You have not authenticated into DataFed Client."
                )
                f.write(
                    "Please follow instructions in the 'Basic Configuration' section in the link below to authenticate yourself:"
                )
                f.write(
                    "https://ornl.github.io/DataFed/user/client/install.html#basic-configuration"
                )

            raise Exception(
                "You have not set up the Globus endpoint. Please follow instructions in the 'Basic Configuration' section in the link below to set up the Globus endpoint: https://ornl.github.io/DataFed/user/client/install.html#basic-configuration"
            )

    @property
    def user_id(self) -> str:
        """
        Gets the user ID from the authenticated user's information.

        Returns:
            str: The user ID extracted from the authenticated user information.
        """
        auth_user = self.getAuthUser()
        if auth_user is None:
            raise RuntimeError("Failed to get authenticated user information")
        return auth_user.split("/")[-1]

    @staticmethod
    def check_string_for_dot_or_slash(s):
        """
        Checks if a string starts with a '.' or '/' and raises an exception if it does.

        Args:
            s (str): The string to check.

        Raises:
            ValueError: If the string starts with either '.' or '/'.
        """
        if s.startswith(".") or s.startswith("/"):
            raise ValueError("String starts with either '.' or '/'")

    @staticmethod
    def find_id_by_title(listing_reply, title_to_find):
        """
        Finds the ID of an item with a specific title from a listing response.

        Args:
            listing_reply (object): The response object containing a list of items.
            title_to_find (str): The title of the item to find.

        Returns:
            str: The DataFed ID of the item with the specified title.

        Raises:
            ValueError: If no item with the specified title is found.
        """
        for item in listing_reply.item:
            if item.title == title_to_find:
                return item.id

        # If no matching title is found, raise an error with a custom message

        raise ValueError(
            f"Project '{title_to_find}' does not exist. "
            "Please create the project and provide an allocation."
        )

    @property
    def get_projects(self, count=500):
        """
        Retrieves a list of projects from DataFed.

        Args:
            count (int, optional): The number of projects to retrieve. Defaults to 500.

        Returns:
            tuple: A tuple containing:
                - list: The list of projects.
                - response: The full response object from the API call.
        """
        response = self.projectList(count=count)
        return response[0], response[1]

    @property
    def getRootColl(self):
        """
        Gets the root collection identifier for the current project.

        Returns:
            str: The root collection identifier formatted with the project ID.
        """
        new_str = self.project_id[:1] + "_" + self.project_id[2:]
        return f"c/{new_str}_root"

    @property
    def _parse_datafed_collection(self):
        """
        Parses the current working directory into components.

        Returns:
            list: A list of directory components split by '/'.
        """
        return self.datafed_path.split("/")

    def getCollList(self, collection_id):
        """
        Retrieves a list of sub-collections within a specified collection.

        Args:
            collection_id (str): The ID of the collection to query.

        Returns:
            tuple: A tuple containing:
                - list: The list of sub-collection titles.
                - ls_resp: The full response object from the API call.
        """
        # Check if the sub-collection exists in DataFed
        ls_resp = self.collectionItemsList(collection_id)

        collections = [record.title for record in ls_resp[0].item]

        return collections, ls_resp

    def create_subfolder_if_not_exists(self):
        """
        Creates sub-folders (collections) if they do not already exist.

        Iterates through the sub-collections specified in `datafed_collection`, creating any that are missing.
        Updates `collection_id` with the ID of the last created or found sub-collection.
        """
        # Gets the root context from the parent collection
        collections, ls_resp = self.getCollList(self.getRootColl)
        current_collection = self.getRootColl

        # Iterate through the sub-collections specified in `datafed_collection`
        for collection in self._parse_datafed_collection[1:]:
            # Check if the collection exists in DataFed
            if collection in collections:
                # Navigate to the sub-collection if it exists
                current_collection = (
                    ls_resp[0]
                    .item[
                        np.where(
                            [record.title == collection for record in ls_resp[0].item]
                        )[0].item()
                    ]
                    .id
                )
            else:
                # Create the sub-collection if it does not exist
                coll_resp = self.collectionCreate(
                    collection, parent_id=current_collection
                )
                current_collection = coll_resp[0].coll[0].id

            # Update the collections list
            collections, ls_resp = self.getCollList(current_collection)

        self.collection_id = current_collection

    def get_notebook_DataFed_ID_from_path_and_title(
        self, notebook_filename, path_id=None
    ):
        """
        Gets the DataFed ID for the Jupyter notebook from the file name and DataFed path

        Args:
            notebook_filename (str): The filename of the notebook. Can be the local filepath or just the filename.

        Returns:
            str: The DataFed ID of the specified notebook

        Raises
            ValueError: If no item with the specified title is found
        """

        ls_resp_2 = self.collectionItemsList(
            path_id if path_id is not None else self.collection_id, count=10000000
        )
        notebook_ID = (
            ls_resp_2[0]
            .item[
                np.where(
                    [
                        record.title == notebook_filename.rsplit("/", 1)[-1]
                        for record in ls_resp_2[0].item
                    ]
                )[0].item()
            ]
            .id
        )

        return notebook_ID

    def data_record_create(
        self,
        metadata=None,
        record_title: Optional[str] = None,
        parent_collection=None,
        deps=None,
        **kwargs,
    ):
        """
        Creates the DataFed record for the saved checkpoint and uploads the relevant metadata

        Args:
            metadata (dict): The relevant model and system metadata for the checkpoint.
            record_title (str): The title of the DataFed record.
            deps (list or str, optional): A list of dependencies or a single dependency to add. Defaults to None.
        Raises:
            Exception: If user is not authenticated or must re-authenticate

        """
        # make sure the Globus endpoint is set
        self.check_if_endpoint_set()
        # make sure the user is logged into DataFed
        self.check_if_logged_in()

        if record_title is None:
            raise ValueError("record_title cannot be None")

        # If the record title is longer than the maximum allowed by DataFed (80 characters)
        # truncate the record title to 80 characters. If logging is true, print out a statement letting the user
        # know the record_title has been truncated.
        if len(record_title) > 80:
            record_title = record_title[:80]  # .replace(".", "_")[:80]
            if self.logging:
                with open(self.log_file_path, "a") as f:
                    timestamp = (
                        datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    f.write(
                        f"\n {timestamp} - Record title is too long. Truncating to 80 characters."
                    )

        # try creating the Data record and uploading the relevant metadata.
        # This will fail when DataFed decides the user must reauthenticate.
        try:
            dc_resp = self.dataCreate(
                str(record_title).rsplit("/", 1)[-1],
                metadata=json.dumps(metadata, cls=UniversalEncoder),
                parent_id=parent_collection
                if parent_collection is not None
                else self.collection_id,
                deps=deps,
                # **kwargs,
            )
            # log that the DataFed data record has been successfully created.
            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n {timestamp} - Data creation successful")

            # return the DataFed listing reply and zip file path
            return dc_resp

        # if the DataFed record creating fails, log the error.
        except Exception as e:
            tb = traceback.format_exc()

            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

                f.write(f"\n {timestamp} - Data creation failed with error: \n {tb}")

            raise e

    def data_record_update(
        self,
        record_id=None,
        record_title: Optional[str] = None,
        metadata=None,
        deps=None,
        overwrite_metadata=False,
        **kwargs,
    ):
        """
        updates the DataFed record for the saved checkpoint including the relevant metadata if it changed

        Args:
            metadata (dict): The relevant model and system metadata for the checkpoint.
            record_title (str): The title of the DataFed record.
            deps (list or str, optional): A list of dependencies or a single dependency to add. Defaults to None.
            overwrite_metadata (bool, default=False): Whether to overwrite the record metadata.
                Merges with existing metadata if false; overwrites if true.
        Raises:
            Exception: If user is not authenticated or must re-authenticate

        """
        # make sure the Globus endpoint is set
        self.check_if_endpoint_set()
        # make sure the user is logged into DataFed
        self.check_if_logged_in()

        if record_title is None:
            raise ValueError("record_title cannot be None")

        # If the record title is longer than the maximum allowed by DataFed (80 characters)
        # truncate the record title to 80 characters. If logging is true, print out a statement letting the user
        # know the record_title has been truncated.
        if len(record_title) > 80:
            record_title = record_title[:80]  # .replace(".", "_")[:80]
            if self.logging:
                with open(self.log_file_path, "a") as f:
                    timestamp = (
                        datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    f.write(
                        f"\n {timestamp} - Record title is too long. Truncating to 80 characters."
                    )

        # try creating the Data record and uploading the relevant metadata.
        # This will fail when DataFed decides the user must reauthenticate.
        try:
            dc_resp = self.dataUpdate(
                record_id,
                title=str(record_title).rsplit("/", 1)[-1],
                metadata=json.dumps(metadata, cls=UniversalEncoder),
                deps_add=deps,
                metadata_set=overwrite_metadata,
                # **kwargs,
            )
            # log that the DataFed data record has been successfully created.
            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n {timestamp} - Data creation successful")

            # return the DataFed listing reply and zip file path
            return dc_resp

        # if the DataFed record creating fails, log the error.
        except Exception as e:
            tb = traceback.format_exc()

            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

                f.write(f"\n {timestamp} - Data creation failed with error: \n {tb}")

            raise e

    @staticmethod
    def addDerivedFrom(deps=None):
        """
        Adds derived from information to the data record, skipping any None values.

        Args:
            deps (list or str, optional): A list of dependencies or a single
                                        dependency to add. Defaults to None.

        Returns:
            list: A list of lists containing the "derived from" information, excluding None entries.
        """
        derived_from_info = []

        # If deps is a string, convert it into a list
        if isinstance(deps, str):
            deps = [deps]

        # If deps is a list, process each entry and skip None entries
        if deps and isinstance(deps, list):
            derived_from_info = [["der", dep] for dep in deps if dep is not None]

        return derived_from_info

    def upload_file(self, DataFed_ID, file_path, wait=False):
        """
        Uploads the file to the DataFed record

        Args:
            DataFed_ID (str): The DataFed ID the data record to upload the file
            file_path (str): The local filepath of the file to upload to DataFed
            wait (bool, optional): whether or not to pause the script until the file has been uploaded. Defaults to False
        """
        # make sure the GLobus enpoint is set
        check_globus_endpoint(self.endpointDefaultGet())

        # try uploading the file.
        try:
            _put_resp = self.dataPut(
                DataFed_ID,
                file_path,
                wait=wait,  # Waits until transfer completes.
            )

            # log that the DataFed data record has been successfully created.
            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

                f.write(
                    "\n This just means that the Data put command ran without errors. \n If the status is not complete, check the DataFed and Globus websites \n to ensure the Globus Endpoint is connected and the file transfer completes."
                )

        # if the DataFed record creating fails, log the error.
        except Exception as e:
            tb = traceback.format_exc()

            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n {timestamp} - Data put failed with error: {tb}")

            raise e

    def getIDs(self, listing_reply):
        """
        Gets the IDs of items from a listing response.

        Args:
            listing_reply (object): The response object containing a list of items.

        Returns:
            list: A list of item IDs.
        """
        return [record.id for record in listing_reply.item]

    def getIDsInCollection(self, collection_id=None):
        """
        Gets the IDs of items in a collection.
        Args:
            collection_id (str): The ID of the collection to query.
        Returns:
            list: A list of item IDs in the collection.
        """
        if collection_id is None:
            collection_id = self.collection_id

        # TODO: make it so it can return more than 10000 records -- not hardcoded
        # Get the list of items in the collection
        collection_list = self.collectionItemsList(collection_id, count=10000)[0]

        # Return the IDs of the items in the collection
        return self.getIDs(collection_list)

    def get_metadata(
        self,
        collection_id=None,
        exclude_metadata=None,
        excluded_keys=None,
        non_unique=None,
        format="pandas",
    ):
        """
        Retrieves the metadata record for a specified record ID.

        Args:
            collection_id (str): The ID of the collection to retrieve metadata from.
            exclude_metadata (str, list, or None, optional): Metadata fields to exclude from the extraction record.
            excluded_keys (str, list, or None, optional): Keys if the metadata record contains to exclude.
            non_unique (str, list, or None, optional): Keys which are expected to be unique independent of record uniqueness - these are not considered when finding unique records.
            format (str, optional): The format to return the metadata in. Defaults to "pandas".

        Returns:
            dict: The metadata record.
        """

        # default condition to get the metadata of the current collection
        if collection_id is None:
            collection_id = self.collection_id

        # Retrieve the data view response for the given record ID
        # TODO: make it so it can return more than 10000 records -- not hardcoded
        _collection_list = self.collectionItemsList(self.collection_id, count=10000)[0]

        # Get the record IDs from the collection list
        record_ids_ = self.getIDsInCollection(collection_id=self.collection_id)

        # Gets a list of Metadata excluding specific metadata terms
        metadata_ = self._get_metadata_list(record_ids_, exclude=exclude_metadata)

        # Exclude specific records if specified key is in the record
        metadata_ = self.exclude_keys(metadata_, excluded_keys)

        if non_unique is not None:
            metadata_ = self.get_unique_dicts(metadata_, exclude_keys=non_unique)

        if format == "pandas":
            import pandas as pd

            self.pd_df = pd.DataFrame(metadata_)

            return pd.DataFrame(metadata_)
        else:
            return ValueError("Invalid format in get_metadata. Please use 'pandas'.")

    def _get_metadata_list(self, record_ids, exclude=None):
        metadata = []
        for record_id in tqdm(record_ids):
            metadata_ = self._get_metadata(record_id)

            if exclude is not None:
                if exclude == "computing":
                    metadata_ = self._remove_computing_metadata(metadata_)
                elif isinstance(exclude, list):
                    metadata_ = self._exclude_metadata_fields(metadata_, exclude)
                else:
                    with open(self.log_file_path, "a") as f:
                        timestamp = (
                            datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                        )

                        f.write(
                            f"\n {timestamp} - Invalid value for exclude parameter in _get_metadata_list."
                        )
                        f.write(
                            "Must be either 'computing' or a list of fields to exclude."
                        )

                    raise ValueError(
                        "Invalid value for exclude parameter in _get_metadata_list. Must be either 'computing' or a list of fields to exclude."
                    )

            metadata.append(metadata_)

        return metadata

    def required_keys(self, dict_list, required_keys):
        """
        Filters a list of dictionaries to include only those that contain all specified required keys.

        Args:
            dict_list (list): A list of dictionaries to filter.
            required_keys (str, list, or set): The keys that each dictionary must contain.
                                               Can be a single string, a list of strings, or a set of strings.

        Returns:
            list: A list of dictionaries that contain all the specified required keys.

        Raises:
            ValueError: If the required_keys parameter is not a string, list of strings, or set of strings.
        """
        # Ensure required_keys is a list, even if a single string is provided
        if isinstance(required_keys, str):
            required_keys = [required_keys]
        elif not isinstance(required_keys, (list, set)):
            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

                f.write(
                    f"\n {timestamp} - Invalid value for required_keys parameter. Must be either a string, list of strings, or set of strings."
                )

            raise ValueError(
                "Invalid value for required_keys parameter. Must be either a string, list of strings, or set of strings."
            )

        # Filter the list of dictionaries to include only those that contain all required keys
        return [d for d in dict_list if all(key in d for key in required_keys)]

    def exclude_keys(self, dict_list, excluded_keys):
        """
        Filters a list of dictionaries to exclude those that contain any of the specified excluded keys.

        Args:
            dict_list (list): A list of dictionaries to filter.
            excluded_keys (str, list, or set): The keys that, if present in a dictionary, will exclude it from the result.
                                            Can be a single string, a list of strings, or a set of strings.

        Returns:
            list: A list of dictionaries that do not contain any of the specified excluded keys.

        Raises:
            ValueError: If the excluded_keys parameter is not a string, list of strings, or set of strings.
        """
        # If excluded_keys is None, return the original list of dictionaries
        if excluded_keys is None:
            return dict_list

        # Ensure excluded_keys is a list, even if a single string is provided
        if isinstance(excluded_keys, str):
            excluded_keys = [excluded_keys]
        elif not isinstance(excluded_keys, (list, set)):
            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

                f.write(
                    f"\n {timestamp} - Invalid value for excluded_keys parameter. Must be either a string, list of strings, or set of strings."
                )

            raise ValueError(
                "Invalid value for excluded_keys parameter. Must be either a string, list of strings, or set of strings."
            )

        # Filter the list of dictionaries to exclude those that contain any of the excluded keys
        return [d for d in dict_list if not any(key in d for key in excluded_keys)]

    @staticmethod
    def get_unique_dicts(dict_list, exclude_keys=None):
        """
        Filters a list of dictionaries to include only unique dictionaries, excluding specified keys.

        Args:
            dict_list (list): A list of dictionaries to filter for uniqueness.
            exclude_keys (list or None, optional): Keys to exclude when determining uniqueness. Defaults to None.

        Returns:
            list: A list of unique dictionaries, excluding specified keys from the uniqueness check.
        """
        if exclude_keys is None:
            exclude_keys = []

        # Convert exclude_keys to a set for efficient lookup
        exclude_keys = set(exclude_keys)

        # List to store unique dictionaries
        unique_dicts = []

        # Set to store hashes of dictionaries excluding exclude_keys
        seen = set()

        def make_hashable(value):
            """
            Recursively convert unhashable types like dicts and lists to hashable types.

            Args:
                value: The value to convert.

            Returns:
                A hashable version of the value.
            """
            if isinstance(value, dict):
                # Convert dict to tuple of key-value pairs
                return tuple((k, make_hashable(v)) for k, v in value.items())
            elif isinstance(value, list):
                # Convert list to a tuple
                return tuple(make_hashable(v) for v in value)
            else:
                # If it's already hashable, return as-is
                return value

        for d in dict_list:
            # Create a tuple that excludes the specified keys from each dictionary
            filtered_items = tuple(
                (k, make_hashable(v)) for k, v in d.items() if k not in exclude_keys
            )

            # If the tuple is not in seen, add it to the unique_dicts
            if filtered_items not in seen:
                seen.add(filtered_items)
                unique_dicts.append(d)

        return unique_dicts

    @staticmethod
    def _exclude_metadata_fields(metadata, fields):
        """
        Excludes specified fields from a metadata dictionary.

        Args:
            metadata (dict): The metadata dictionary to exclude fields from.
            fields (list): A list of fields to exclude from the metadata.

        Returns:
            dict: A dictionary with the specified fields excluded.
        """
        # Use dictionary comprehension to create a new dictionary excluding specified fields
        return {key: value for key, value in metadata.items() if key not in fields}

    def _remove_computing_metadata(
        self, metadata, fields=["gpu", "optimizer", "cpu", "memory", "python", "layers"]
    ):
        """
        Removes computing-related metadata fields from the metadata dictionary.

        Args:
            metadata (dict): The metadata dictionary to remove fields from.
            fields (list, optional): A list of fields to remove from the metadata.
                                     Defaults to ['gpu', 'optimizer', 'cpu', 'memory', 'python', 'layers'].

        Returns:
            dict: A dictionary with the specified fields removed.
        """
        # Use the _exclude_metadata_fields method to remove the specified fields from the metadata
        return self._exclude_metadata_fields(metadata, fields)

    @staticmethod
    def _extract_metadata_fields(metadata, fields):
        """
        Extracts specified fields from a metadata dictionary.

        Args:
            metadata (dict): The metadata dictionary to extract fields from.
            fields (list): A list of fields to extract from the metadata.

        Returns:
            dict: A dictionary containing only the specified fields from the metadata.
        """
        return {field: metadata[field] for field in fields}

    def _get_metadata(self, record_id):
        """
        Retrieves the metadata for a specified record ID.

        Args:
            record_id (str): The ID of the record to retrieve metadata for.

        Returns:
            dict: A dictionary containing the metadata of the specified record,
                  including the record ID.
        """
        # Retrieve the data view response for the given record ID
        dv_resp = self.dataView(record_id)

        # Parse the metadata from the response and convert it to a dictionary
        dict_ = json.loads(dv_resp[0].data[0].metadata)

        # Add the record ID to the dictionary
        dict_["id"] = dv_resp[0].data[0].id

        return dict_

    def check_no_files(self, record_ids):
        """
        Checks if any of the specified DataFed records have no associated files.

        Args:
            record_ids (list): A list of DataFed record IDs to check.

        Returns:
            list or None: A list of record IDs that have no associated files, or None if all records have files.
        """
        no_files = []
        for record_id in tqdm(record_ids):
            # Check if the record has no associated files by checking the size attribute
            if self.dataView(record_id)[0].data[0].size == 0:
                no_files.append(record_id)

        # Return None if all records have files, otherwise return the list of record IDs with no files
        if no_files == []:
            return None
        else:
            return no_files

    def replace_missing_records(
        self, collection_id=None, file_path=None, upload_kwargs=None, logging=True
    ):
        if upload_kwargs is not None:
            kwargs = self.upload_kwargs.copy()
            kwargs.update(upload_kwargs)
        else:
            kwargs = self.upload_kwargs

        if collection_id is None:
            collection_id = self.collection_id

        if logging:
            print(f"checking collection {collection_id} for missing records")

        missing_record_ids = self.check_no_files(
            self.getIDsInCollection(collection_id=collection_id)
        )

        if missing_record_ids is not None:
            if logging:
                print(f"found {len(missing_record_ids)} missing records")

            if logging:
                print("retrieving metadata for missing records")

            metadata = self._get_metadata_list(missing_record_ids)

            for i, (record_id, metadata) in enumerate(
                zip(missing_record_ids, metadata)
            ):
                if logging:
                    if "Model Parameters" in metadata.keys():  # record is a checkpoint
                        print(
                            f"trying to reupload {metadata['Model Parameters']['filename']} for record {record_id}"
                        )

                        if self.check_if_file_data(
                            metadata["Model Parameters"]["filename"],
                            metadata["Model Parameters"]["path"],
                        ):
                            self.upload_file(
                                record_id,
                                self.joinPath(
                                    metadata["Model Parameters"]["filename"],
                                    metadata["Model Parameters"]["path"],
                                ),
                                wait=kwargs.get("wait", False),
                            )
                    elif "script" in metadata.keys():  # record is notebook
                        print(
                            f"trying to reupload {metadata['script']['path']} for record {record_id}"
                        )
                        if self.check_if_file_data(
                            metadata["script"]["path"], file_path
                        ):
                            self.upload_file(
                                record_id,
                                metadata["script"]["path"],
                                wait=kwargs.get("wait", False),
                            )

    def getFileName(self, record_id):
        """
        Retrieves the file name (without extension) associated with a record ID.

        Args:
            record_id (str): The ID of the record to retrieve the file name for.

        Returns:
            str: The file name without the extension.
        """
        # Get the source path of the file associated with the record
        source_path = self.dataView(record_id)[0].data[0].source

        # Extract the file name from the source path and remove the extension
        file_name = source_path.split("/")[-1]

        return file_name

    def getRecordTitle(self, record_id):
        """
        Retrieves the title of a record from its ID.

        Args:
            record_id (str): The ID of the record to retrieve the title for.

        Returns:
            str: The title of the record.
        """
        return self.dataView(record_id)[0].data[0].title

    def getFileExtension(self):
        """
        Retrieves the file extension of the dataset file.

        Returns:
            str: The file extension of the dataset file, including the leading dot.
        """
        # Split the file name by '.' and return the last part as the extension
        return "." + self.getFileName(self.dataset_id_or_path).split(".")[-1]

    def getData(self, dataset_id: Optional[str] = None):
        """
        Downloads the data from the dataset
        """

        self.data_path = None  # Bandaid fix; this needs to be defined for the class!

        if dataset_id is None:
            if self.dataset_id_or_path is None:
                raise ValueError("dataset_id_or_path is not set")
            dataset_id = self.dataset_id_or_path

        # if a data path is not provided, download the data to the current directory
        if self.data_path is None:
            self.dataGet(dataset_id, "./", **self.download_kwargs)
        else:
            file_name = self.getFileName(dataset_id)

            # if the data path does not exist, create it
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)

            if not self.check_if_file_data(file_name):
                if os.path.exists(
                    os.path.join(
                        self.data_path, dataset_id[2:] + self.getFileExtension()
                    )
                ):
                    file_name = dataset_id[2:] + self.getFileExtension()
                else:
                    print(
                        f"Downloading {dataset_id} data using datafed to {self.data_path}"
                    )
                    self.dataGet(dataset_id, self.data_path, **self.download_kwargs)

            self.file_path = self.joinPath(file_name)

    def joinPath(self, file_name, path_name=None):
        """
        Joins the data path and the file name to create a full file path.

        Args:
            file_name (str): The name of the file.
            path_name (str,default=None): The name of the file path. Defaults to self.local_model_path

        Returns:
            str: The full file path.
        """
        if path_name is None:
            return os.path.join(self.local_model_path, file_name)
        else:
            return os.path.join(path_name, file_name)

    def check_if_file_data(self, file_name, path_name=None):
        """
        Check if a file exists in the specified data path.

        Args:
            file_name (str): The name of the file to check.

        Returns:
            bool: True if the file exists in the data path, False otherwise.

        """
        if path_name is None:
            if os.path.exists(self.joinPath(file_name)):
                return True
            else:
                return False
        else:
            if os.path.exists(self.joinPath(file_name, path_name)):
                return True
            else:
                return False
