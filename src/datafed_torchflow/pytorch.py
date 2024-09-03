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


class TorchLogger(nn.Module):
    
    def __init__(self, DataFed_path, model_name_base, isRecord = True):
        super(TorchLogger, self).__init__()
        self.model_name = model_name_base + '.pt' # what is this????? 
        
        self.DataFed_instance = DataFed(DataFed_path) #datafed_path
        
        self.df_api = API()
        self.DataFed_Path = DataFed_path
        self.isRecord = isRecord

        # TODO: write a function to extract project name and searches the filename hierarchy for the collection.
        ## project/collection --> create if not exists -- get ID.
        ## project/collection/ collection --> create if not exists -- get ID.
        
        
        
    def get_DataFed_ID_from_path(self):
    
        DataFed_path_list = self.DataFed_instance._parse_cwd
        
        
        try: 
            
            for idx, elem in enumerate(DataFed_path_list[1:]):
                
                if idx == 0: 
                    ls_resp = self.df_api.collectionItemsList(f"c/p_{DataFed_path_list[0].lower()}_root")
                    
                else: 
                    ls_resp = self.df_api.collectionItemsList(DataFed_parent_collection_id)                
                
                
                if ls_resp[0].item == []:
                    trials_already_in_DataFed = []
                    if idx == 0: 
                        DataFed_parent_collection_id = f"c/p_{DataFed_path_list[0].lower()}_root"
                    # otherwise use the same one from the previous iteration of the loop. 
                    
                    # just create the record
                    
                
                else: # collection not empty 
                    DataFed_parent_collection_id = ls_resp[0].item[idx].id 
                    trials_already_in_DataFed = []
                    for record in ls_resp[0].item:
                        trials_already_in_DataFed.append(record.title)
                        
                if elem in trials_already_in_DataFed:
                    #coll_resp = folder_model
                    DataFed_parent_collection_id = ls_resp[0].item[np.where(elem in trials_already_in_DataFed)[0].item()].id
                    ls_resp_2 = self.df_api.collectionItemsList(DataFed_parent_collection_id)
                    #Datafed_collection_name = folder_model

                
                else: #create collection or record if path at end and is for a record
                    if idx == len(DataFed_path_list)-2 and self.isRecord == True: 
                        #rec_resp = self.df_api.dataCreate("/".join(DataFed_path_list[0:idx+1]),metadata = json.dumsp({"testing": 123}), parent_id=DataFed_parent_collection_id) # ADD DATA/METADATA STUFF??????? 
                        rec_resp = self.df_api.dataCreate(elem,metadata = json.dumps({"testing": 123}), parent_id=DataFed_parent_collection_id) # ADD DATA/METADATA STUFF??????? 

                        ls_resp_2 = self.df_api.collectionItemsList(coll_resp[0].coll[0].id)
                        DataFed_parent_collection_id = rec_resp[0].data[0].id
                        
                    else: 
                       # coll_resp = self.df_api.collectionCreate("/".join(DataFed_path_list[0:idx+1]),parent_id=DataFed_parent_collection_id)
                        coll_resp = self.df_api.collectionCreate(elem,parent_id=DataFed_parent_collection_id)
                        ls_resp_2 = self.df_api.collectionItemsList(coll_resp[0].coll[0].id)
                        DataFed_parent_collection_id = coll_resp[0].coll[0].id

                    
                        
                if ls_resp_2[0].total == 10000:

                    new_collections_made+=1
                    #coll_resp = self.df_api.collectionCreate(f"{'/'.join(DataFed_path_list[0:idx+1])}_{new_collections_made}",
                    coll_resp = self.df_api.collectionCreate(f"{elem}_{new_collections_made}",
                                            parent_id=DataFed_parent_collection_id
                                            )
                    DataFed_parent_collection_id = coll_resp[0].coll[0].id
                    
                if idx == len(DataFed_path_list)-2:
                    return DataFed_parent_collection_id
            
        except Exception as e:
            print("Exception:", e)
            print("*"*20)
            print("Traceback:",traceback.format_exc())
            print("*"*20)
            print("Project does not exist")
            print("Manually create the project and try again")        
    
        
    def save_state(self):
        torch.save(self.state_dict(), self.model_name)
        
    def save_model(self):
        torch.save(self, self.model_name)
        
        




def upload_to_datafed(folder_model, log_file_path, globus_endpoint_set, DataFed_project_name, 
                      DataFed_record_title, DataFed_record_metadata, DataFed_record_filename):
    
    
    """
    Uploads model results to DataFed with metadata, handles file organization, and logs the upload status.

    Parameters:
        folder_model (str): Name of the folder containing the model's weights.
        log_file_path (str): Path where the progress and error message logs are recorded.
        globus_endpoint_set (bool): Indicates whether the Globus endpoint is set and active for the data transfer. 
        globus_endpoint_set (bool): Indicates whether the Globus endpoint is set and active for the data transfer.  
        DataFed_project_name (str): Identifier of the DataFed project where data is stored.
        DataFed_record_title (str): Title for the new DataFed record.
        DataFed_record_metadata (dict): Metadata to be associated with the new DataFed record.
        DataFed_record_filename (str): File containing the data to be uploaded.

    Returns:
        None: Progress and error messages are logged in file `log_file_path`. If `Globus_endpoint_set` is True,
        a DataFed record with title `DataFed_record_title`, metadata `DataFed_record_metadata` and 
        data `DataFed_record_filename` are uploaded to DataFed under project `DataFed_project_name`.  
    """

    
    try: 
        

        
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

            
                
        if ls_resp_2[0].total == 10000:

            new_collections_made+=1
            coll_resp = df_api.collectionCreate(f"{folder_model}_{new_collections_made}",
                                    parent_id=DataFed_project_name
                                    )
            Datafed_collection_id = coll_resp[0].coll[0].id


                
        
    except:
        tb = traceback.format_exc()
        with open(log_file_path,"a") as f:
            timestamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

            f.write(f'\n {timestamp} - Failed to navigate to DataFed repository with error: \n {tb}')

    try:
        
 
        dc_resp = df_api.dataCreate(DataFed_record_title, 
                                    metadata = json.dumps(DataFed_record_metadata),parent_id = Datafed_collection_id)
        
           
        with open(log_file_path, "a") as f:
            timestamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

            f.write(f'\n {timestamp} - Data creation sucessful')
        


    except Exception:
        tb = traceback.format_exc()

        with open(log_file_path, "a") as f:
            timestamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

            f.write(f'\n {timestamp} - Data creation failed with error: \n {tb}')
        
    try:
        if globus_endpoint_set and df_api.endpointDefaultGet(): 
            
    
            
            put_resp = df_api.dataPut(dc_resp[0].data[0].id,
                                    str(DataFed_record_filename),
                                    wait=False,  # Waits until transfer completes.
                                    )            
            
            
            with open(log_file_path, "a") as f:
                
                
                
                
                current_task_status = put_resp[0].task.msg 
                
                timestamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

                f.write(f'\n {timestamp} - Data put status: {current_task_status}')
                f.write(f'\n This just means that the Data put command ran without errors. \n If the status is not complete, check the DataFed and Globus websites \n to ensure the Globus Endpoint is connected and the file transfer completes.')
                
        
                
                
        else:
            globus_endpoint_set = False

            with open(log_file_path, "a") as f:
                timestamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f'\n {timestamp} - Globus endpoint not set \n Please set your Globus endpoint and try again')
            


    except Exception:
        tb = traceback.format_exc()
        
        with open(log_file_path, "a") as f:
            timestamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

            f.write(f'\n {timestamp} - Data put failed with error: {tb}')