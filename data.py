""" This script includes everything related to the cardio dataset e.g
   - The corresponding DataSet class definition
   - Preprocessing until the point its ready to be loaded into a DataLoader
   - Splitting
   - Helper functions. 

    Synthetic data postprocessing (mainly reverting to original domain) functions
    are also in this file. """
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
import torch
from pandas.core.algorithms import isin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Pytorch dataset definition used in both main and pretrain for Cardio data"""
class Dataset:
    # !!! NOTE: The GAN training dataset should also contain the y-variable cardio, because we want to synthesize it as well.
    def __init__(self, data, device):
        
        if isinstance(data, pd.DataFrame): #Check if @data is pd.dataframe and convert to numpy for pytorch
            self.data = data.to_numpy()

        self.device = device
        self.sampleSize = data.shape[0] 
        self.featureSize = data.shape[1] 

    def return_data(self): #return data, unaltered
        return data

    def __len__(self):
        return len(self.data)

    #idx are the indexes passed to dataset by dataloader to create batches.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.data[idx]
        sample = (torch.from_numpy(sample)).float().to(self.device) #convert to torch and assign to a device.
        return sample #return a batch


"""Preprocessing for Cardio dataset"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Used values: 
#   Systolic blood pressure | Examination Feature | ap_hi | int | max 180 systolic can be deadly 
#   Diastolic blood pressure | Examination Feature | ap_lo | int | diastolic max 120 / hypertensive crisis
#   Anything lower than 90/60 mm Hg is low blood pressure (hypotension). 
#   Set bounds for blood pressure at realistic values according to sources-
#   ap_lo 40/140, which is -20 and +20 over the worst cases 
#   ap_hi 60/200, which is -20 points and + 20
#   Height to 120, 210 which is more than +3 -3 standard deviations away 
#   Weight to 35, since all are adults no upper cut which is more than +3 -3 standard deviations away 
#-------------------------------------------------------------------------------------------------------------
# if you want to make changes, change hard-coded values inside the functions below

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Reads cardio data"""
def read_cardio_data(cardio_data_path):
    df = pd.read_csv(cardio_data_path, sep=";", 
    usecols= ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio'])
    return(df)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""KNN imputation for all features of a dataset, replace pd.na values"""
def knn_imputation(df):
    feature_names = df.columns.values
    imputer = KNNImputer(n_neighbors = 3)
    imputer.fit(df)
    df = imputer.transform(df)
    df = pd.DataFrame(df, columns = feature_names)
    return(df)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Treat outliers, bounds set in function, outside values put to pd.na and knn-imputed"""
def cardio_treat_outliers(df):
    # --- Outliers (outside range) are put to pd.na
    age_bool_mask = (df['age'] < 14000)
    height_bool_mask = (df['height'] < 120) | (df['height'] > 210)
    weight_bool_mask = (df['weight'] < 35) 
    ap_lo_bool_mask = (df['ap_lo'] < 40) | (df['ap_lo'] > 140)
    ap_hi_bool_mask = (df['ap_hi'] < 60) | (df['ap_hi'] > 200)

    df['age'] = df['age'].mask(age_bool_mask) #values that are out of range are marked as pd.na
    df['height'] = df['height'].mask(height_bool_mask)
    df['weight'] = df['weight'].mask(weight_bool_mask)
    df['ap_lo'] = df['ap_lo'].mask(ap_lo_bool_mask)
    df['ap_hi'] = df['ap_hi'].mask(ap_hi_bool_mask)

    # ---  #knn imputation for pd.na values
    df = knn_imputation(df)
    return df 

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Min-max scaling"""
def min_max(df, MINMAX_LOW, MINMAX_HIGH):
    feature_names = df.columns.values
    scaler = MinMaxScaler(feature_range = (MINMAX_LOW,MINMAX_HIGH))
    df = scaler.fit(df).transform(df)
    df = pd.DataFrame(df, columns = feature_names)
    return df

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Create GAN training data, no splits and all features"""
def cardio_preprocess_gan_training_data(cardio_path):
    df = read_cardio_data(cardio_path)
    df = cardio_treat_outliers(df)
    df = pd.get_dummies(data = df, columns = ['cholesterol', 'gluc']) 
    df = df.rename(columns = {"cholesterol_1.0":"cholesterol_1", "cholesterol_2.0":"cholesterol_2","gluc_1.0":"gluc_1","gluc_2.0":"gluc_2", "gluc_3.0":"gluc_3"}) #this fixed pd.get_dummies stupid float naming
    df['gender'] = df['gender'].replace({1:0, 2:1}) # Fix gender, original: 1 women 2 men - AFTER: 0 women, 1 men
    df = min_max(df, -1, 1)
    return df

#%%
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Create classification data (preprocessed, minmaxed to [0,1])"""
def cardio_preprocess_classification_data(cardio_path):
    df = read_cardio_data(cardio_path)
    df = cardio_treat_outliers(df)
    df = pd.get_dummies(data = df, columns = ['cholesterol', 'gluc'], dtype = int) 
    df = df.rename(columns = {"cholesterol_1.0":"cholesterol_1", "cholesterol_2.0":"cholesterol_2", "cholesterol_3.0":"cholesterol_3","gluc_1.0":"gluc_1","gluc_2.0":"gluc_2", "gluc_3.0":"gluc_3"}) #this fixed pd.get_dummies stupid float naming
    df['gender'] = df['gender'].replace({1:0, 2:1}) # Fix gender, original: 1 women 2 men - AFTER: 0 women, 1 men
    df = min_max(df, 0, 1)
    return df

""" Stratified split for data, read from csv
@split: 1 - split determines training data size, in other words, split is the size of the set a val/test set is made from
For example: 0.5 would split to train and test, and then test is split to 0.25, 0.25"""
def stratified_split(y_feature, split, split_seed, path):
    df = pd.read_csv(path)

    # - A stratified split. Size of test dataset is 1 - split
    train_df, test_df = train_test_split(df, test_size = 1-split, stratify = df[y_feature], random_state = split_seed)
    val_df, test_df = train_test_split(test_df, test_size = 0.5, stratify = test_df[y_feature], random_state = split_seed)
    return train_df, val_df, test_df 

def separate_to_X_and_Y(df, y_name):
    X, Y = df.drop(y_name, 1), df[y_name]
    return (X, Y)
#%%

# ---------------------------CREATE AND SAVE DIFFERENT DATASETS MADE OUT OF CARDIO----------------------------
""" The difference between gan training data and classification data is, that gan training data is -1,1 for 
TanH to work while logreg is 0,1 minmaxed to make it compatible with scikit-learn logistic regression."""

"""Create and save gan training data in folder, no splits, all features, minmaxed to -1, 1"""
#gan_training_data = cardio_preprocess_gan_training_data("/home/vajnie/thesis/erikoistyo/data/cardio/cardio_original.csv")
#gan_training_data.to_csv("./data/cardio/training/cardio_gan_train.csv", index = False)
#
"""Create and save logistic regression training data in folder, no splits, all features, minmaxed to 0,1"""
# The split is done using @cardio_stratified_split inside evaluation, so that seed can be changed during running it
#classification_data = cardio_preprocess_classification_data("/home/vajnie/thesis/erikoistyo/data/cardio/cardio_original.csv")
#classification_data.to_csv("./data/cardio/evaluation/classification_data.csv", index = False)#

#%%
""" Generate synthetic data from a generator and post-process (round)
    synth data values are rounded to closest full"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Synth test is not returned now as it is not needed, but its included in code to make split code 
# as close as possible to real data code and subset size. 
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

""" Same method as above, but returns the whole synth data without splitting"""
def generate_synth_data_no_split(netG, noise, feature_names):
    
    #Generate
    synth_data = netG(noise).detach()
    synth_data = pd.DataFrame(synth_data.numpy(), columns = feature_names)
    
    #Return synth train X, Y and synth val X, Y
    return synth_data
