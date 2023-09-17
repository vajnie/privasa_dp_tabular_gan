import argparse
import pickle
import os
from my_utils import mkdir

RESULT_DIR = '' # Set the directory to save your trained models


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_models_dir_path', '-saved_models_dir_path', type=str, default= '.', help= 'where models are saved')
    parser.add_argument('--random_seed', '-s', type=int, default=1, help='random seed')
    parser.add_argument('--splits', '-split', type=float, default=0.9, help='training set size - what portion of data should be used for training? The rest will be evenly divided to validation and test sets. If splist = 0, then will use absolute size 60 000 and 10 000 as in MNIST')
    parser.add_argument('--dataset', '-data', type=str, default='cardio', choices=['cardio'],
                        help=' dataset name')
    parser.add_argument('--num_discriminators', '-ndis', type=int, default=250, help='number of discriminators')
    parser.add_argument('--noise_multiplier', '-noise', type=float, default=1.07, help='noise multiplier')
    parser.add_argument('--z_dim', '-zdim', type=int, default=32, help='latent code dimensionality')
    parser.add_argument('--model_dim', '-mdim', type=int, default=64, help='model dimensionality')
    parser.add_argument('--n_features_out', '-n_features_out', type=int, default=16, help='number of features the generator should produce when creating tabular data when variables are one hotted, only used when the TabularGenerator is used')
    parser.add_argument('--batchsize', '-bs', type=int, default=32, help='batch size')
    parser.add_argument('--L_gp', '-lgp', type=float, default=10, help='gradient penalty lambda hyperparameter')
    parser.add_argument('--L_epsilon', '-lep', type=float, default=0.001, help='epsilon penalty (used in PGGAN)')
    parser.add_argument('--critic_iters', '-diters', type=int, default=5, help='number of critic iters per gen iter')
    parser.add_argument('--latent_type', '-latent', type=str, default='bernoulli', choices=['normal', 'bernoulli'],
                        help='latent distribution')
    parser.add_argument('--iterations', '-iters', type=int, default=20000, help='iterations for training')
    parser.add_argument('--pretrain_iterations', '-piters', type=int, default=2000, help='iterations for pre-training')
    parser.add_argument('--num_workers', '-nwork', type=int, default=0, help='number of workers')
    parser.add_argument('--net_ids', '-ids', type=int, nargs='+', help='the index list for the discriminator')
    parser.add_argument('--print_step', '-pstep', type=int, default=100, help='number of steps to print')
    parser.add_argument('--vis_step', '-vstep', type=int, default=1000, help='number of steps to vis & eval')
    parser.add_argument('--save_step', '-sstep', type=int, default=5000, help='number of steps to save')
    parser.add_argument('--load_dir', '-ldir', type=str, help='checkpoint dir (for loading pre-trained models)')
    parser.add_argument('--pretrain', action='store_true', default=False, help='if performing pre-training')
    parser.add_argument('--num_gpus', '-ngpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--gen_arch', '-gen', type=str, default='my_generator', choices=['my_generator'],
                        help='generator architecture')
    parser.add_argument('--run', '-run', type=int, default=1, help='index number of run')
    parser.add_argument('--exp_name', '-name', type=str,
                        help='output folder name; will be automatically generated if not specified')
    args = parser.parse_args()
    return args


def save_config(args):
    '''
    store the config and set up result dir
    :param args:
    :return:
    '''
    ### set up experiment name
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
            print(kv_str)
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
