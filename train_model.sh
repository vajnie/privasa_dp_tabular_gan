#!/usr/bin/env bash

### This bash script runs pretrain and training for a single set of configurations.

#############################################################
### RUN ARGUMENTS 
#############################################################
meta_start=0 # the discriminator start index for the current process (need to be modified for each process)
ndis=250  # total number of discriminators to train 
seed=20
gen_training_iterations_in_main=40000

#-------PATHS------------------------------------------------------------------
exp_name= '' #name experiment
ldir= '' #load dir, where the pretrained discriminators are loaded from. 
result_file_name= '' #results file directory name. 

##################################
### Pretrain discriminators
##################################
#start=$((meta_start))
#end=$((start + ndis - 1))
#vals=$(seq $start $end)
#python my_pretrain.py -data 'cardio' -ids $vals --pretrain -gen 'my_generator' -name 'cardio' -noise 0. -ndis $ndis -name $exp_name

##################################
### Train the main model
##################################
python main.py -name $exp_name -ndis $ndis -ldir $ldir -s $seed -iters $gen_training_iterations_in_main


