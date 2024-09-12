import numpy as np
from datafed.CommandLib import API

class DataFed(API):
	
    def __init__(self, cwd, verbose=False):
        super(API, self).__init__()
        self.cwd = cwd
        self.verbose = verbose
        self.df_api = API()
        
        self.check_if_logged_in()
        self.check_if_endpoint_set()
        
        # checks if the cwd is a valid path
        self.check_string_for_dot_or_slash(self.cwd)
        
        items, response = self.get_projects
        
        # checks if the project exists in DataFed
        self.project_id = self.find_id_by_title(items, self._parse_cwd[0])
        
    def check_if_logged_in(self):   
        if self.df_api.getAuthUser():
            if self.verbose:
                print("Success! You have been authenticated into DataFed as: " + self.df_api.getAuthUser())
        else:
            raise Exception("You have not authenticated into DataFed Client. Please follow instructions in the 'Basic Configuration' section in the link below to authenticate yourself: https://ornl.github.io/DataFed/user/client/install.html#basic-configuration")
        
    def check_if_endpoint_set(self):
        if self.df_api.endpointDefaultGet():
            if self.verbose:
                print(f"Success! You have set up the Globus endpoint {self.df_api.endpointDefaultGet()}.")
        else:
            raise Exception("You have not set up the Globus endpoint. Please follow instructions in the 'Basic Configuration' section in the link below to set up the Globus endpoint: https://ornl.github.io/DataFed/user/client/install.html#basic-configuration")
    
    def get_collection_id(self):
        # main function that navigates through the DataFed project and collections to find the collection ID
        
        # Function to parse the DataFed project
        df_paths = self._parse_cwd()
        
        # Check if the project exists
    
    @staticmethod
    def check_string_for_dot_or_slash(s):
        if s.startswith('.') or s.startswith('/'):
            raise ValueError("String starts with either '.' or '/'")
    
    @staticmethod    
    def find_id_by_title(listing_reply, title_to_find):
        for item in listing_reply.item:  
            if item.title == title_to_find:
                return item.id

        # If no matching title is found, raise an error with a custom message
        raise ValueError(f"Project '{title_to_find}' does not exist. "
                        "Please create the project and provide an allocation.")
        
    @property
    def get_projects(self, count=500):
        response = self.df_api.projectList(count=count)
        return response[0], response[1]
    
    @property    
    def _parse_cwd(self):
        return self.cwd.split('/')
    
        # Function to parse the current working directory and return the project and collection names
    
    def check_if_project_exists(self):
        # Function to check if the project exists in DataFed
        pass
    
    
    def create_subfolder_if_not_exits(self,DataFed_collection_name, DataFed_subcollection_name):
        # check if the sub-collection exists in DataFed    
        ls_resp = self.df_api.collectionItemsList(DataFed_collection_name)

        trials_already_in_DataFed = []
        for record in ls_resp[0].item:
            trials_already_in_DataFed.append(record.title)
            
        if DataFed_subcollection_name in trials_already_in_DataFed:
            #navigate to the sub-collection if exists
            Datafed_collection_id = ls_resp[0].item[np.where(DataFed_subcollection_name in trials_already_in_DataFed)[0].item()].id
           # ls_resp_2 = self.df_api.collectionItemsList(Datafed_collection_id) ## WE DON'T CARE ABOUT THE CONTENTS OF THE COLLECTION, SO DON'T NEED THIS LINE 
            #Datafed_collection_name = folder_model

            
        else:
            # create sub-collection if doesn't exist
            coll_resp = self.df_api.collectionCreate(DataFed_subcollection_name,parent_id=DataFed_collection_name)
            #ls_resp_2 = self.df_api.collectionItemsList(coll_resp[0].coll[0].id) ## WE DON'T CARE ABOUT THE CONTENTS OF THE COLLECTION, SO DON'T NEED THIS LINE
            Datafed_collection_id = coll_resp[0].coll[0].id
            
        return Datafed_collection_id