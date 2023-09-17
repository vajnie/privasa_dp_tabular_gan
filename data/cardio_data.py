#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#This script includes everything that depends specifically on the cardio dataset
# These are:
#   - The corresponding DataSet class definition
#   - Preprocessing until the point its ready to be loaded into a DataLoader
#   - Splitting
#   - Helper functions. 
# @Author: Valtteri Nieminen, 27.10.2021, last update 05.05.2022
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# load_cardio_data gives the whole data preprocessed. This is used in main.py and pretrain.py
# load_cardio_data_args gives the data with args seed and split. Split determines division to train/test for downstream experiments. 


import pandas as pd
import torch

#---------------------------
# Used in main and pretrain!
#----------------------------
class Dataset:
    # !!! NOTE !!!
    # Dataset ALSO CONTAINS THE Y-VARIABLE CARDIO, because we want to synthesize it as well. 
    #@type: if numpy is default, if you want pandas specify type = pd
    def __init__(self, data, device, convert_to_np = True):
        
        #Convert to numpy
        if(convert_to_np):
            if isinstance(data, pd.DataFrame):
                self.data = data.to_numpy()

        #Dont convert to numpy       
        else:
            self.data = data

        self.device = device
        self.sampleSize = data.shape[0] 
        self.featureSize = data.shape[1] 

    def return_data(self):
        return data

    def __len__(self):
        return len(self.data)

    #idx are the indexes passed to dataset by dataloader
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.data[idx]
        sample = (torch.from_numpy(sample)).float().to(self.device)
        y = sample[9]
        return sample, y