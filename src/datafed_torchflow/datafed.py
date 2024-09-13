import numpy as np
from datafed.CommandLib import API


class DataFed(API):
    def __init__(self, cwd, verbose=False):
        super().__init__()
        self.cwd = cwd
        self.verbose = verbose

        self.check_if_logged_in()
        self.check_if_endpoint_set()

        # checks if the cwd is a valid path
        self.check_string_for_dot_or_slash(self.cwd)

        # checks if user is saving in the root collection
        if self._parse_cwd[0] == self.user_id:
            self.project_id = self.user_id
        else:
            # gets all the projects in DataFed
            items, response = self.get_projects

            # checks if the project exists in DataFed
            self.project_id = self.find_id_by_title(items, self._parse_cwd[0])

    def check_if_logged_in(self):
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
        return self.getAuthUser().split("/")[-1]

    def check_if_endpoint_set(self):
        if self.endpointDefaultGet():
            if self.verbose:
                print(
                    f"Success! You have set up the Globus endpoint {self.endpointDefaultGet()}."
                )
        else:
            raise Exception(
                "You have not set up the Globus endpoint. Please follow instructions in the 'Basic Configuration' section in the link below to set up the Globus endpoint: https://ornl.github.io/DataFed/user/client/install.html#basic-configuration"
            )

    # def get_collection_id(self):
    #     # main function that navigates through the DataFed project and collections to find the collection ID

    #     # Function to parse the DataFed project
    #     df_paths = self._parse_cwd()

    #     # Check if the project exists

    @staticmethod
    def check_string_for_dot_or_slash(s):
        if s.startswith(".") or s.startswith("/"):
            raise ValueError("String starts with either '.' or '/'")

    @staticmethod
    def find_id_by_title(listing_reply, title_to_find):
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
        response = self.projectList(count=count)
        return response[0], response[1]

    @property
    def getRootColl(self):
        new_str = self.project_id[:1] + "_" + self.project_id[2:]
        return f"c/{new_str}_root"

    @property
    def _parse_cwd(self):
        return self.cwd.split("/")

    def getCollList(self, collection_id):
        # check if the sub-collection exists in DataFed
        ls_resp = self.collectionItemsList(collection_id)

        collections = []
        for record in ls_resp[0].item:
            collections.append(record.title)

        return collections, ls_resp

    def create_subfolder_if_not_exits(self):
        # gets the root context from the parent collection
        collections, ls_resp = self.getCollList(self.getRootColl)
        current_collection = self.getRootColl

        # iterate through the sub-collections
        for collection in self._parse_cwd[1:]:
            # check if the collection exists in DataFed
            if collection in collections:
                # navigate to the sub-collection if exists
                current_collection = (
                    ls_resp[0].item[np.where(collection in collections)[0].item()].id
                )

            else:
                # create sub-collection if doesn't exist
                coll_resp = self.collectionCreate(
                    collection, parent_id=current_collection
                )
                current_collection = coll_resp[0].coll[0].id

            # update the collections list
            collections, ls_resp = self.getCollList(current_collection)

        self.collection_id = current_collection