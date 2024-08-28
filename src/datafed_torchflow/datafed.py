from datafed.CommandLib import API

class DataFed(API):
	
    def __init__(self, cwd):
        super(API, self).__init__()
        self.cwd = cwd
    
    def get_collection_id(self):
        # main function that navigates through the DataFed project and collections to find the collection ID
        
        # Function to parse the DataFed project
        df_paths = self._parse_cwd()
        
        # Check if the project exists
        
        
    
    @property    
    def _parse_cwd(self):
        return self.cwd.split('/')
    
        # Function to parse the current working directory and return the project and collection names
    
    def check_if_collection_exists(self):
        # Function to check if the project exists in DataFed
        pass
    
    
    
        ls_resp = df_api.collectionItemsList(DataFed_project_name)

        trials_already_in_DataFed = []
        for record in ls_resp[0].item:
            trials_already_in_DataFed.append(record.title)
            
        if folder_model in trials_already_in_DataFed:
            #coll_resp = folder_model
            Datafed_collection_id = ls_resp[0].item[np.where(folder_model in trials_already_in_DataFed)[0].item()].id
            ls_resp_2 = df_api.collectionItemsList(Datafed_collection_id)
            #Datafed_collection_name = folder_model

            
        else:
            coll_resp = df_api.collectionCreate(folder_model,parent_id=DataFed_project_name)
            ls_resp_2 = df_api.collectionItemsList(coll_resp[0].coll[0].id)
            Datafed_collection_id = coll_resp[0].coll[0].id