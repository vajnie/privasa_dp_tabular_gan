# ---- This is a new script to replace the cardio_data.py script
# - Load and split the presaved, preprocessed gan_training_set_data. 
#-------------------------------------------------------------------

#%%
import sys
import pandas as pd
from my_models import * 
from my_utils import*
from sklearn.model_selection import train_test_split

def load_cardio_stratified_split(train_set_size, split_seed, path):
    df = pd.read_csv(path)

    # - A stratified split. Size of test dataset is 1 - train_set_size
    train_df, test_df = train_test_split(df, test_size = 1-train_set_size, stratify = df['cardio'], random_state = split_seed)
    val_df, test_df = train_test_split(test_df, test_size = 0.5, stratify = test_df['cardio'], random_state = split_seed)
    return train_df, val_df, test_df 


#Synth test is not returned now as it is not needed, but its included in code to make split code as close as possible. 
def generate_synth_data(netG, noise, train_set_size, feature_names, columns_to_round, split_seed):

    #Generate
    synth_data = netG(noise).detach()
    synth_data = pd.DataFrame(synth_data.numpy(), columns = feature_names)
    #Postprocess
    
    for col in columns_to_round:
        synth_data[col] = synth_data[col].round()  #round values to closest full 

    #Stratified split
    synth_train, synth_val = train_test_split(synth_data, test_size = 1- train_set_size, stratify = synth_data['cardio'], random_state = split_seed)
    synth_val, synth_test = train_test_split(synth_val, test_size = 0.5, stratify = synth_val['cardio'], random_state = split_seed)

    synthetic_train_X, synthetic_train_Y = synth_train.drop('cardio', 1), synth_train['cardio']
    synthetic_val_X, synthetic_val_Y = synth_val.drop('cardio', 1), synth_val['cardio']
    
    #Return synth train X, Y and synth val X, Y
    return synthetic_train_X, synthetic_train_Y, synthetic_val_X, synthetic_val_Y

def generate_synth_data_no_split(netG, noise, feature_names, columns_to_round):

    #Generate
    synth_data = netG(noise).detach()
    synth_data = pd.DataFrame(synth_data.numpy(), columns = feature_names)
    #Postprocess
    
    for col in columns_to_round:
        synth_data[col] = synth_data[col].round()  #round values to closest full 

    #Return synth train X, Y and synth val X, Y
    return synth_data


#%%