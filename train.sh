#!/usr/bin/env bash
# This script runs main training of the (dp) generator or dp generators, see further documentation in pretrain.sh
# the usage of this script is the same. 

#-------- Change arguments to suit your pre-training settings ------------


data='./mock_data/mock_data.csv'  # same data that was used in pre-training, and will be used for the training of main gen
ndis=1  # total number of discriminators that were used in pretraining.  
ldir='./results/pretrain/my_experiment' #load dir, where the pretrained discriminators are loaded


seed=4
gen_training_iterations_in_main=1
exp_name='my_experiment' #name of experiment for the saving directory
result_file_name='my_test_results' #results file directory name. 

#--------------------------------------TRAIN MAIN MODEL----------------------------------------------------
python main.py -name $exp_name -ndis $ndis -ldir $ldir -seed $seed -iters $gen_training_iterations_in_main -data $data

