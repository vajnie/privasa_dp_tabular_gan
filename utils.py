import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sklearn as sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
import scipy.stats as ss


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


#@input model - a reference to the modeling function (not the object). For example LogisticRegression.
#@hyperparameters list - list of different hyperparameter sets as in @Class Hyperparameter_searchspace.get_n_random_hyperparameter_sets(n)
# Used to run multiple models, searching for best hyperparameter settings.

def run_search(model, hyperparams_list, X_train, X_dev, X_test, Y_train, Y_dev, Y_test):
    best_s, best_hyperparams = -np.Inf, None
    
    for i, hyperparams in enumerate(hyperparams_list):
        print("On sample %d / %d (hyperparams = %s)" % (i+1, len(hyperparams_list), repr((hyperparams))))
        M = model(**hyperparams, ) #unpack keyword arguments from the hyperparameters dictionary
        M.fit(X_train, Y_train) 
        s = roc_auc_score(Y_dev, M.predict_proba(X_dev)[:, 1]) #test on dev set
        
        #Update best hyperparam.
        if s > best_s:
            best_s, best_hyperparams = s, hyperparams
            print("New Best Score: %.2f @ hyperparams = %s" % (100*best_s, repr((best_hyperparams))))
    
    #We do not concatenate train and dev because of leaks, but we might if its only real data and this can be implemented later. 
    
    return run_single_model(model, best_hyperparams, X_train, X_test, Y_train, Y_test)


""" Hyperparameter_searchspace is a dictionary of hyperparameters that 
we choose randomly from using the Choice class.
@input: a dictionary of hyperparameters, see example usage
@output: a searchspace object that can be used to get random sets of hyperparameter settings. 

### Example usage ###
Hyperparameter_searchspace({
    'C': Choice(np.geomspace(1e-3, 1e3, 10000)),
    'penalty': Choice(['l1', 'l2']), and so on. 
    The choice here would be made uniformly. 
    """
class Hyperparameter_searchspace():
    def __init__(self, dict_of_hyperparameters): self.dict_of_hyperparameters = dict_of_hyperparameters
        
    def get_n_random_hyperparameter_sets(self, n):
        #Make the random choices for dictionary elements. 
        a = {k: v.random_choice(n) for k, v in self.dict_of_hyperparameters.items()}  
        out = []
        for i in range(n): out.append({k: vs[i] for k, vs in a.items()})
        return out
"""
the choice class makes a random, uniform selection between elements in a list.
# for example in the dictionary of logistic regression hyperparameters we could have one key-value pair as follows:
# 'solver': Choice(['liblinear', 'saga']) and here there would be a random choice between the two solvers. 
"""
class Choice():
    def __init__(self, options): self.options = options
    def random_choice(self, n): return [self.options[i] for i in ss.randint(0, len(self.options)).rvs(n)]
    
### Define hyperparameter search space ###    
# set the hyperparameters we make choices and random sets over
logistic_regression_hyperparameters = Hyperparameter_searchspace({
    'C': Choice(np.geomspace(1e-3, 1e3, 10000)),
    'penalty': Choice(['l1', 'l2']),
    'solver': Choice(['liblinear', 'saga']),
    'max_iter': Choice([100, 500])
})


### Run a SINGLE model and calculate auc, auprc, acc and F1 scores on it
#@input model - a reference to the modeling function (not the object). For example LogisticRegression.
#@hyperparameters - dictionary, a set as in @Class Hyperparameter_searchspace
# Used to run the final model with best hyperparams found, X_train here would be train + dev set.
def run_single_model(model, hyperparameters, X_train, X_test, Y_train, y_test):
    m = model(**hyperparameters)
    m.fit(X_train, Y_train)
    y_true  = y_test
    y_score = m.predict_proba(X_test)[:, 1]
    y_pred  = m.predict(X_test)

    auc   = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    acc   = accuracy_score(y_true, y_pred)
    F1    = f1_score(y_true, y_pred)
    
    return m, hyperparameters, auc, auprc, acc, F1, y_pred


def round_numbers(number):
    rounded_num = int(round(number))
    if((rounded_num) < 0):
        rounded_num == 0
    elif((rounded_num) > 1): 
        rounded_num == 1
    return rounded_num



# As in: generally [x,y] -> [a,b] 
# Where: [a,b] is the target
# x, y are set as defaults
def reverse_min_max(b, a, value_xy, x = -1, y = 1):
    return( ((value_xy - x) / (y - x)) * (b - a) + a )


""" This is a helper function for training the generator during pretraining. If we make a infinite loop, we dont have to 
specify a maximum for gen training iterations that depend on pretraining length and so we avoid specifying an extra argument. 
"""
def inf_train_gen(trainloader):
    while True:
        for data in trainloader:
            yield (data)


""" Getter that helps assigning the pretrained discriminators to device during main training"""
def get_device_id(id, num_discriminators, num_gpus):
    partitions = np.linspace(0, 1, num_gpus, endpoint=False)[1:]
    device_id = 0
    for p in partitions:
        if id <= num_discriminators * p:
            break
        device_id += 1
    return device_id            