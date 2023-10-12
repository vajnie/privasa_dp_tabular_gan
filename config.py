import argparse
import pickle
import os
from utils import *

# ----------------- SET NAME OF FOLDER UNDER WHICH YOU WANT ALL RESULTS TO GO -------------------
# This folder will be created if there isnt one, otherwise subfolders created according to settings
# in pretrain.sh and train.sh 

RESULT_DIR = './results'
# -----------------------------------------------------------------------------------------------


""" Parse arguments given to for ex main.py. When main.py starts it will first parse default arguments.
If it gets after that a argument from the commandline it will use that instead.""" 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', '-save_dir', type=str, default= '/home/vajnie/projects/gradu/gan_models/wgan_gp_private/results/cardio', help= 'where models are saved')
    parser.add_argument('--random_seed', '-seed', type=int, default=1, help='random seed')
    parser.add_argument('--splits', '-split', type=float, default=0.9, help='training set size - what portion of data should be used for training classification? The rest will be evenly divided to validation and test sets. If splist = 0, then will use absolute size 60 000 and 10 000 as in MNIST')
    parser.add_argument('--data_path', '-data', type=str, help='dataset path')
    parser.add_argument('--num_discriminators', '-ndis', type=int, default=250, help='number of discriminators')
    parser.add_argument('--noise_multiplier', '-noise', type=float, default=1.07, help='noise multiplier')
    parser.add_argument('--delta', '-delta', type=float, default=1e-5, help='delta for DP')
    parser.add_argument('--z_dim', '-zdim', type=int, default=32, help='latent code dimensionality')
    parser.add_argument('--model_dim', '-mdim', type=int, default=64, help='model hidden layer dimensionality')
    parser.add_argument('--batchsize', '-bs', type=int, default=32, help='batch size')
    parser.add_argument('--critic_iters', '-diters', type=int, default=5, help='number of critic iters per gen iter')
    parser.add_argument('--iterations', '-iters', type=int, default=20000, help='iterations for training')
    parser.add_argument('--pretrain_iterations', '-piters', type=int, default=2000, help='iterations for pre-training')
    parser.add_argument('--net_ids', '-ids', type=int, nargs='+', help='the index list for the discriminator')
    parser.add_argument('--load_dir', '-ldir', type=str, help='checkpoint dir (for loading pre-trained models)')
    parser.add_argument('--pretrain', action='store_true', default=False, help='should be given as -- argument if performing pre-training')
    parser.add_argument('--run', '-run', type=int, default=1, help='index number of run')
    parser.add_argument('--exp_name', '-name', type=str, help='output folder name; will be automatically generated if not specified')
    parser.add_argument('--num_workers', '-nwork', type=int, default=0, help='number of workers for pytorch dataloader, this just has to be here, but its always default value that is used')
    parser.add_argument('--L_gp', '-lgp', type=float, default=10, help='gradient penalty lambda hyperparameter')
    parser.add_argument('--experiment_folder_path', '-expfolderpath', type=str, help='folder that contains models to evaluate during evaluate.py')
    parser.add_argument('--num_gpus', '-ngpus', type=int, default=1, help='number of gpus to use, default 1')
    parser.add_argument('--n_features', '-n_features', type=int, default=2, help='number of features to generate')
    parser.add_argument('--n_output_sample', '-n_output_sample', type=int, default=1000, help='number of samples to sample from generators')
    parser.add_argument('--raw_data_path', '-raw_data_path', type=str, default= '', help= 'path to raw data, used to get min and max of features when creating datasets')
    parser.add_argument('--specific_checkpoint_save_iters', '-specific_checkpoint_save_iters', type=list, default= [15,249,550,949,1449,4988], help= 'list of iterations one can fill for checkpoints to be saved at on top of the usual checkpoint save between checkpoint_save_interval. Default is at epsilon 1,2,3,4,5,10 when subsampling rate is 1/500')
    parser.add_argument('--checkpoint_save_interval', '-checkpoint_save_interval', type=int, default= 100, help= 'interval to save checkpoint models at')
    parser.add_argument('--debug_prints', '-debug_prints', default=False, help= 'gradient modification hook debug prints')
    
    args = parser.parse_args()
    return args

  
def save_config(args):
    '''
    store the config and set up result dir
    :param args:
    :return:
    '''
    ### set up experiment name if one not given
    if args.exp_name is None:
        exp_name = '{}_Ndis{}_Noise{}_Zdim{}_Mdim{}_BS{}_Lgp{}_Lep{}_Diters{}_{}_Run{}'.format(
            args.gen_arch,
            args.num_discriminators,
            args.noise_multiplier,
            args.z_dim,
            args.model_dim,
            args.batchsize,
            args.L_gp,
            args.L_epsilon,
            args.critic_iters,
            args.latent_type,
            args.run)
        args.exp_name = exp_name

    if args.pretrain:
        save_dir = os.path.join(RESULT_DIR, 'pretrain', args.exp_name)
    else:
        save_dir = os.path.join(RESULT_DIR, 'main', args.exp_name)
    args.save_dir = save_dir

    ### save config
    mkdir(save_dir)
    config = vars(args)
    pickle.dump(config, open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in config.items():
            kv_str = k + ':' + str(v) + '\n'
            #print(kv_str)
            f.writelines(kv_str)


def load_config(args):
    '''
    load the config
    :param args:
    :return:
    '''
    assert args.exp_name is not None, "Please specify the experiment name"
    if args.pretrain:
        save_dir = os.path.join(RESULT_DIR,  'pretrain', args.exp_name)
    else:
        save_dir = os.path.join(RESULT_DIR,  'main', args.exp_name)
    assert os.path.exists(save_dir)

    ### load config
    config = pickle.load(open(os.path.join(save_dir, 'params.pkl'), 'rb'))
    return config
