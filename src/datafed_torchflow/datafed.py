from datetime import datetime
import traceback
import numpy as np
from datafed.CommandLib import API
import json
from m3util.globus.globus import check_globus_endpoint
from datafed_torchflow.JSON import UniversalEncoder
from tqdm import tqdm


class DataFed(API):
    """
    A class to interact with DataFed API.

    Inherits from:
        API: The base class for interacting with the DataFed API.

    Attributes:
        cwd (str): The current working directory.
        verbose (bool): Flag to enable verbose logging.
        project_id (str): The ID of the project.
    """

    def __init__(self, cwd, verbose=False, log_file_path=".log.txt"):
        """
        Initializes the DataFed instance.

        Args:
            cwd (str): The current working directory.
            verbose (bool, optional): Flag to enable verbose logging. Defaults to False.

        Raises:
            Exception: If the user is not authenticated with DataFed.
        """
        super().__init__()
        self.cwd = cwd
        self.verbose = verbose

        self.check_if_logged_in()
        self.check_if_endpoint_set()

        # Checks if the cwd is a valid path.
        self.check_string_for_dot_or_slash(self.cwd)

        # Set the log file path
        self.log_file_path = log_file_path

        # Checks if user is saving in the root collection.
        if self._parse_cwd[0] == self.user_id:
            self.project_id = self.user_id
        else:
            # Gets all the projects in DataFed.
            items, response = self.get_projects

            # Checks if the project exists in DataFed.
            self.project_id = self.find_id_by_title(items, self._parse_cwd[0])

            self.create_subfolder_if_not_exists()

    def check_if_logged_in(self):
        """
        Checks if the user is authenticated with DataFed.

        Raises:
            Exception: If the user is not authenticated.
        """
        if self.getAuthUser():
            if self.verbose:
                print(
                    "Success! You have been authenticated into DataFed as: "
                    + self.getAuthUser()
                )
        else:
            raise Exception(
                "You have not authenticated into DataFed Client. Please follow instructions in the 'Basic Configuration' section in the link below to authenticate yourself: https://ornl.github.io/DataFed/user/client/install.html#basic-configuration"
            )

    @property
    def user_id(self):
        """
        Gets the user ID from the authenticated user's information.

        Returns:
            str: The user ID extracted from the authenticated user information.
        """
        return self.getAuthUser().split("/")[-1]

    def check_if_endpoint_set(self):
        """
        Checks if the Globus endpoint is set up.

        Raises:
            Exception: If the Globus endpoint is not set up.
        """
        if self.endpointDefaultGet():
            if self.verbose:
                print(
                    f"Success! You have set up the Globus endpoint {self.endpointDefaultGet()}."
                )
        else:
            raise Exception(
                "You have not set up the Globus endpoint. Please follow instructions in the 'Basic Configuration' section in the link below to set up the Globus endpoint: https://ornl.github.io/DataFed/user/client/install.html#basic-configuration"
            )

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
            str: The ID of the item with the specified title.

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
    def _parse_cwd(self):
        """
        Parses the current working directory into components.

        Returns:
            list: A list of directory components split by '/'.
        """
        return self.cwd.split("/")

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

        Iterates through the sub-collections specified in `cwd`, creating any that are missing.
        Updates `collection_id` with the ID of the last created or found sub-collection.
        """
        # Gets the root context from the parent collection
        collections, ls_resp = self.getCollList(self.getRootColl)
        current_collection = self.getRootColl

        # Iterate through the sub-collections specified in `cwd`
        for collection in self._parse_cwd[1:]:
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

    def data_record_create(self, metadata, record_title, deps=None, **kwargs):
        self.check_if_endpoint_set()
        self.check_if_logged_in()

        if len(record_title) > 80:
            record_title = record_title.replace(".", "_")[:80]
            if self.verbose:
                print("Record title is too long. Truncating to 80 characters.")

        try:
            dc_resp = self.dataCreate(
                record_title.replace(".", "_"),
                metadata=json.dumps(metadata, cls=UniversalEncoder),
                parent_id=self.collection_id,
                deps=deps,
                **kwargs,
            )

            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

                f.write(f"\n {timestamp} - Data creation successful")

            return dc_resp

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

    def upload_file(self, dc_resp, file_path, wait=False):
        check_globus_endpoint(self.endpointDefaultGet())

        try:
            put_resp = self.dataPut(
                dc_resp[0].data[0].id,
                file_path,
                wait=wait,  # Waits until transfer completes.
            )

            with open(self.log_file_path, "a") as f:
                current_task_status = put_resp[0].task.msg

                timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

                f.write(f"\n {timestamp} - Data put status: {current_task_status}")
                f.write(
                    "\n This just means that the Data put command ran without errors. \n If the status is not complete, check the DataFed and Globus websites \n to ensure the Globus Endpoint is connected and the file transfer completes."
                )

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

    def get_metadata(self, record_id, format="pandas"):
        """
        Retrieves the metadata record for a specified record ID.

        Args:
            record_id (str): The ID of the record to retrieve.

        Returns:
            dict: The metadata record.
        """

        collection_list = self.collectionItemsList(self.collection_id, count=10000)[0]
        record_ids = self.getIDs(collection_list)

        return self.dataGet(record_id)[0].data[0].metadata

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
                    raise ValueError(
                        "Invalid value for exclude parameter. Must be either 'computing' or a list of fields to exclude."
                    )

            metadata.append(metadata_)

        return metadata

    @staticmethod
    def required_keys(dict_list, required_keys):
        """
        Checks if a list of dictionaries contains the required keys.

        Args:
            dict_list (list): A list of dictionaries to check.
            required_keys (list): A list of keys that must be present in each dictionary.

        Returns:
            bool: True if all dictionaries contain the required keys, False otherwise.
        """
        return all(all(key in d for key in required_keys) for d in dict_list)

    @staticmethod
    def excluded_keys(dict_list, excluded_keys):
        """
        Checks if a list of dictionaries contains the excluded keys.

        Args:
            dict_list (list): A list of dictionaries to check.
            excluded_keys (list): A list of keys that must not be present in any dictionary.

        Returns:
            bool: True if no dictionary contains the excluded keys, False otherwise.
        """
        return all(all(key not in d for key in excluded_keys) for d in dict_list)

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
