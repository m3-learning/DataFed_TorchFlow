import numpy as np
from datafed.CommandLib import API

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

    def __init__(self, cwd, verbose=False):
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

        # Checks if user is saving in the root collection.
        if self._parse_cwd[0] == self.user_id:
            self.project_id = self.user_id
        else:
            # Gets all the projects in DataFed.
            items, response = self.get_projects

            # Checks if the project exists in DataFed.
            self.project_id = self.find_id_by_title(items, self._parse_cwd[0])

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
