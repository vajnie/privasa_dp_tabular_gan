#!/usr/bin/env bash
# This script is used to run pretraining with user-specified arguments.
# A call is made to pretrain.py, arguments can be given to it by changing them in this script, 
# non-specified arguments will be @config.py default arguments.  
#------------------------------------------------------------------------------------------------- 
# INPUT: a STRING path, @pretrain_args, to a text file, that contains user specified settings listed below: 
# $meta_start: starting index for discriminator filenames. For ex., 0 results in discriminator.pth files named netD_0 ... netD_ndis. 
# $ndis: number of discriminators to train (and number of splits to data).
# $exp_name: the name of the experiment, that will be used to create a folder for the results.
#-----------------------------------USE---------------------------------------------------------
# Change the arguments below. Arguments come by default from config.py and you can give any arguments listed there 
# as input to this script to change them. 
# meta_start should be set to 0 unless you want to do training on different machines or continue from some point. 


meta_start=0 # starting index for discriminators, default 0. # forex., 0 will results in discriminator.pth filenames netD_0 ... netD_ndis. 
data='mock_data/mock_data.csv' #path to gan training data
ndis=1 #number of discriminators or "subsampling rate"
exp_name='my_experiment'
piters=10 #pretrain iterations
seed=10

# --- Call pretrain --------------------------------------------------------------------------------------------------------------------
start=$((meta_start))
end=$((start + ndis - 1))
vals=$(seq $start $end)
python pretrain.py -data $data -ids $vals --pretrain -noise 0. -ndis $ndis -name $exp_name -seed $seed -piters $piters
