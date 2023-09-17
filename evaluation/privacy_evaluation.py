import sys
from autodp import rdp_acct, rdp_bank

#Add previous folder to sys.path to find my_config
sys.path.insert(0, './../model')

from my_config import *

def main(config):
    delta = 1e-5
    batch_size = config['batchsize']
    prob = 1. / config['num_discriminators']  # subsampling rate
    n_steps = config['iterations']  # training iterations
    sigma = config['noise_multiplier']  # noise scale
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    acct = rdp_acct.anaRDPacct()
    acct.compose_subsampled_mechanism(func, prob, coeff=n_steps * batch_size)
    epsilon = acct.get_eps(delta)
    print("Privacy cost is: epsilon={}, delta={}".format(epsilon, delta))


if __name__ == '__main__':
    args = parse_arguments()
    main(load_config(args))



def evaluate_epsilon(iterations, batchsize, sigma, prob, delta):
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
    acct = rdp_acct.anaRDPacct()
    acct.compose_subsampled_mechanism(func, prob, coeff= iterations* batchsize)
    epsilon = acct.get_eps(delta)

    #print("Privacy cost is: epsilon={}, delta={}".format(epsilon, delta))
    return epsilon
    
#%%
