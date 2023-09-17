0###################################################################################################
# Evaluate ONE generator accross all its checkpoints (model state at different iterations).
###################################################################################################
# args: seed (int), generator save folder path (string)
# example command to run in bash:
# <python evaluation_pipeline.py 10 "/home/main/model_folder">
###################################################################################################
# OVERVIEW:


#---------- 0. Setup namespace and load generators (checkpoints) of the run into dict -------------
# @gendict = {checkpoint_path, iterations}
# checkpoint here refers to a model saved after some amount of iterations. 

#----------------- 1. Privacy analysis on all checkpoints -------------------------------------

#------------------ 2. Run downstream quality evaluation  -------------------------------------      
# Get best hyperparameters and AUC for all checkpoints.
# Save in quality_evaluation_results = pd.DataFrame(columns = checkpoint_name, [used_hyperparams], AUC)
# privacy analysis results are also concat. into this df.
##################################################################################################
#--------
#%% --- 0. Setup ---------
#--- Imports and additional paths
import sys
import os
from os import mkdir

import pandas as pd
import numpy as np
import torch
import re 
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import ParameterGrid
from autodp import rdp_acct, rdp_bank
from my_models import *
from sklearn.model_selection import train_test_split


#--- Evaluation scripts
from privacy_evaluation import evaluate_epsilon
from load_cardio_data import load_cardio_stratified_split
from load_cardio_data import generate_synth_data

#--------------------------------------------------------------------
# 0. Setup NAMESPACE and LOAD generators of the run into a dictionary 
#---------------------------------------------------------------------
#@generators is a dictionary of int iterations and string generator file path
#@params is the parameters used to train the model

# Arguments, paths, global variables
#-------------------------------------
delta = 1e-5
DATA_PATH = "/home/data/log_reg_data.csv" #log reg data should be minmaxed to [0,1], gan last layer is sigmoid after training to match this. 

SEED = int(sys.argv[1]) # first argument, [0] is always script name, second - [1] - is the fir+st given argument. 
path_to_run_folder = sys.argv[2] # path to run folder appended with "/" 

path_to_params = path_to_run_folder + "/params.txt" #params file contains used parameters for this run.  
checkpoint_file_names = (os.listdir(path_to_run_folder + "/intermediate")) #here are the checkpoints
final_model_file_name = "netG.pth" #this is always the same, but its not in the intermediate folder where the chekcpoint models are. 

#Load generator params into dictionary
#-------------------------------------
params = pd.read_csv(path_to_params, delimiter = ":", names = ["param", "value"])
params = dict(zip(params['param'], params['value']))
exp_name = params['exp_name'] #experiment name, given when training models. 
split_ratio = float(params['splits']) #this is the ratio of training/test data. 
columns_to_round = [] # discrete variables of your dataset
# for example with the cardio data discrete columns would be rounded: 
# columns_to_round = ['gender','smoke', 'alco', 'active', 'cardio', 'cholesterol_1', 'cholesterol_2', 'cholesterol_3','gluc_1', 'gluc_2', 'gluc_3']

### Create the @Generators dict = {iterations (string) : generator_file_paths (string)}
#---------------------------------------------------------------------------
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
checkpoint_file_names.sort(key = natural_keys)

#Make strings of the iterations in the names to use as dict keys
numbers_in_checkpoint_file_names = [ re.findall("[0-9]", checkpoint_file_name) for checkpoint_file_name in checkpoint_file_names]
iteration_strings = [''.join(numbers) for numbers in numbers_in_checkpoint_file_names]
iteration_strings.append(str(int(iteration_strings[-1]) + 999)) #make the last name to be +1000, this is the model that has run for max_iterations

#Make paths for the different generators in order to load them 
generator_file_paths = [ (path_to_run_folder + "/intermediate/" + checkpoint_file_name) for  checkpoint_file_name in checkpoint_file_names ] #these are in same order as iteration_strings.
generator_file_paths.append(path_to_run_folder + "/netG.pth")
generators = {iteration_strings[i]: generator_file_paths[i] for i in range(len(iteration_strings))} #Create dict = {iteration_strings: generator_file_paths}

#%% Create a directory for results
#-------------------------------
result_folder_name = ""
path_to_results_folder = f"{path_to_run_folder}/{result_folder_name}"
try:
    mkdir(path_to_results_folder)
except OSError as error:
    print(f"Folder {path_to_results_folder} already exists, no new folder created.") 

# ---------------2. Privacy analysis on checkpoints ---------------------------

# Get generator parameters used in privacy evaluation
#-----------------------------------------------------
privacy_keys = ["batchsize", "num_discriminators", "noise_multiplier"]
batchsize, num_discriminators, sigma = [params.get(key) for key in privacy_keys] 
batchsize, num_discriminators, sigma = int(batchsize), int(num_discriminators), float(sigma)
prob = 1. / num_discriminators

# Privacy analysis across all checkpoints of different iterations
# @privacy_results = tuple( int:iterations, epsilon)
#----------------------------------------------------------------

iterations, epsilons = [], []
for n_iterations in generators.keys():
    cur_iterations = int(n_iterations)
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
    acct = rdp_acct.anaRDPacct()
    acct.compose_subsampled_mechanism(func, prob, coeff= cur_iterations* batchsize)
    epsilon = acct.get_eps(delta)
    #print("Privacy cost is: epsilon={}, delta={}".format(epsilon, delta))
    
    iterations.append(cur_iterations)
    epsilons.append(epsilon)
    


privacy_results = pd.DataFrame({"iterations" : iterations, "epsilon" : epsilons})
privacy_results.to_csv(f"{path_to_results_folder}" + "/privacy_results.csv")

#TODO: SAVE ALL RESULTS IN A NICE CSV FOR THIS ONE GENERATOR AND DIFFERENT ITERATION AMOUNTS with best auc etc. 
#results_file_name = "privacy_evaluation_" + exp_name + ".csv"

# ----------------3. Run downstream quality evaluation -----------------------------

# Load data, below would be the data loading case with the cardio dataset
# For some other dataset you need
#------------------------------------------------------------------------------------
target_variable = 'cardio'
real_train_split_with_Y, real_val_split_with_Y, real_test_split_with_Y = load_cardio_stratified_split(0.8, SEED, DATA_PATH) 
real_train_X, real_train_Y = real_train_split_with_Y.drop(target_variable, 1), real_train_split_with_Y[target_variable]
real_val_X, real_val_Y = real_val_split_with_Y.drop(target_variable, 1), real_val_split_with_Y[target_variable]
real_test_X, real_test_Y = real_test_split_with_Y.drop(target_variable, 1), real_test_split_with_Y[target_variable]

real_data_total_size = real_train_split_with_Y.shape[0] + real_val_split_with_Y.shape[0] + real_test_split_with_Y.shape[0] 
#%%
colnames = ["C", "penalty", "solver", "AUC", "case"]
checkpoint_result_column_names = colnames + ['iterations', 'epsilon', 'delta']
checkpoint_result_summary = pd.DataFrame(columns = checkpoint_result_column_names)    
#@ DataFrame: results_summary is a dataframe where different checkpoints results will be appended. 

#%%

# Evaluation loop
# ----------------

for num_iterations, checkpoint_path in generators.items():

    # Load trained generator 
    #-----------------------
    checkpoint_name = checkpoint_path.split("/")[-1] #get name of checkpoint from path 
    z_dim = int(params.get('z_dim')) #latent vector length
    n_features_out = int(params.get('n_features_out'))
    model_dim = int(params.get('model_dim')) 
    device = 'cpu' 
    feature_names = real_train_split_with_Y.columns.values

    netG = Generator(z_dim = z_dim, hidden_dim = model_dim, n_features_out= n_features_out).to(device) #empty model
    netG.load_state_dict(torch.load(checkpoint_path)) #load weights from path specified in conf. (save path) 
    netG.gen[3][1] = nn.Sigmoid() #change last layer to sigmoid so we dont have to bring to same scale with data. 

    print("Checkpoint_path", checkpoint_path)

    # Sample synthetic training and synthetic validation datasets FOR ONE GENERATOR CHECKPOINT
    #-----------------------------------------------------------------------------------------

    z_noise = get_noise(real_data_total_size, z_dim, device)
    synthetic_train_X, synthetic_train_Y, synthetic_val_X, synthetic_val_Y = generate_synth_data(netG, z_noise, 0.8, feature_names, columns_to_round, SEED)
    
    synth_train_y0, synth_train_y1, = synthetic_train_Y.value_counts()  
    synth_val_y0, synth_val_y1, = synthetic_val_Y.value_counts()
    real_train_y0, real_train_y1, = real_train_Y.value_counts()  
    real_val_y0, real_val_y1, = real_val_Y.value_counts()
    real_test_y0, real_test_y1, = real_test_Y.value_counts()


    size_info = {"synth_train_y0" : synth_train_y0, "synth_train_y1" : synth_train_y1, "synth_val_y0" : synth_val_y0, "synth_val_y1" : synth_val_y1, "real_train_y0" : real_train_y0, "real_train_y1" : real_train_y1, "real_val_y0" : real_val_y0, "real_val_y1" : real_val_y1, "real_test_y0" : real_test_y0, "real_test_y1" : real_test_y1}
    print(size_info)

    with open(f"{path_to_results_folder}/size_info.csv", 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in size_info.items():
            writer.writerow([key, value])
    #size_info['generator'] = checkpoint_name
    #size_info.to_csv(path_to_results_folder + "/y_distributions.csv")


#%%
    #%%
    #1.1 Grid search a logistic regression (LR) parameter space and make sure edges are not hit
    #-------------------------------------------------------------------------------------------
    parameter_search_space_dict = ({
        'C': np.logspace(-4, 0.15,20), #base 10 log so 10^4 biggest 
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    })

    result_df = pd.DataFrame(columns = colnames)
    best_score = 0.
    best_scores = []
    best_grids = []

    for _, g in enumerate(ParameterGrid(parameter_search_space_dict)):
        model = LogisticRegression(**g) #give current params to model
        model.fit(synthetic_train_X, synthetic_train_Y)
        auc = roc_auc_score(synthetic_val_Y, model.predict_proba(synthetic_val_X)[:, 1])
        result_values = list(g.values())
        result_values.append(auc)
        result_values.append("syn_v_syn_val")
        result_series = pd.Series(result_values, index = result_df.columns)
        result_df = result_df.append(result_series, ignore_index= True)

        # save if best
        if auc > best_score:
            print("New best auc syn v syn: ", auc, "with parameters:", g, "checkpoint:", num_iterations, "path:", checkpoint_path)
            best_score = auc
            best_scores.append(best_score)
            best_hyperparameters = g
            best_grids.append(g) #these will be evaluated on the real data. 


    #------- Evaluate AUC with a set of best hyperparameters ---------

    synthetic_final_train_X = pd.concat([synthetic_train_X, synthetic_val_X], axis = 0) #make a set of both train and val syn data
    synthetic_final_train_Y = pd.concat([synthetic_train_Y, synthetic_val_Y])

    #----------------------------------------------------------------------
    epsilon = evaluate_epsilon(int(num_iterations), batchsize, sigma, prob, delta)
    model = LogisticRegression(**best_hyperparameters)
    model.fit(synthetic_final_train_X, synthetic_final_train_Y) 
    final_auc = roc_auc_score(real_test_Y, model.predict_proba(real_test_X)[:, 1]) 
    checkpoint_result_values = list(best_hyperparameters.values()) + [final_auc, 'syn_v_real_best_unbiased', num_iterations, epsilon, delta]
    print("VALUES: ", checkpoint_result_values)
    checkpoint_result_values = pd.Series(checkpoint_result_values, index = checkpoint_result_summary.columns)
    checkpoint_result_summary = checkpoint_result_summary.append(checkpoint_result_values, ignore_index= True)
    print(checkpoint_result_summary)

    #--------- The output checkpoint_result_summary, syn_v_real_best is the unbiased reported AUC --------
    checkpoint_result_summary.to_csv(path_to_results_folder + f"/seed_{SEED}_checkpoint_result_summary.csv", header = checkpoint_result_column_names, index = False)



