import os, sys
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from config import *
from models import *
from utils import * 
from data import *

#--------------------------------------------------------------------------------------------------------------------------------------------------------
# This pre-training procedure and script is originally based on the work and code of Chen, Orekondy and Fritz https://github.com/DingfanChen/GS-WGAN
# and the paper of the aforementioned authors GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators, 2020-12-06, NeurIPS
# The code has been extensively modified to accommodate different architectures and data types, mainly tabular data. 
#--------------------------------------------------------------------------------------------------------------------------------------------------------#


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SETUP 

def main(args):

# --- Load arguments from config.py or from given arguments ----    
    data_path = args.data_path
    num_discriminators = args.num_discriminators
    z_dim = args.z_dim
    model_dim = args.model_dim
    batchsize = args.batchsize
    critic_iters = args.critic_iters #critic, or discriminator iterations per gen iter
    save_dir = args.save_dir
    net_ids = args.net_ids 
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    L_gp = args.L_gp #gradient penalty lambda parameter, default 10 
    split = args.splits #determines the train/test split, default is 0.9, resulting in 0.9*df.shape[0] training set size


# --- Set up Cuda, or device available ---
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

# --- Set up Data ---
    train_df = pd.read_csv(data_path) 
    trainset = Dataset(train_df, device)
    # Get parameters dependent on data
    n_features = train_df.shape[1]

# --- Initialize discriminators and put to a list ---
    netD_list = [] 
    for _ in range(len(net_ids)): 
        netD = Critic(model_dim, n_features)
        netD_list.append(netD)
    netD_list = [netD.to(device) for netD in netD_list]

    optimizerD_list = [] # pytorch optimizers for discr.
    for i in range(len(net_ids)):
        netD = netD_list[i]
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerD_list.append(optimizerD)


# --- Create indexing files for invividual discriminators ---
    # (these are indices in training data related to each discriminator)
    if os.path.exists(os.path.join(save_dir, 'indices.npy')):
        print('Indices loaded from disk')
        indices_full = np.load(os.path.join(save_dir, 'indices.npy'), allow_pickle=True)
    else:
        print('New data discriminator subset index files created')
        indices_full = np.arange(len(trainset))
        np.random.shuffle(indices_full)
        indices_full.dump(os.path.join(save_dir, 'indices.npy'))
    trainset_size = int(len(trainset) / num_discriminators)
    

# --- Input pipelines for individual discriminators ---
    input_pipelines = []
    for i in net_ids:
        start = i * trainset_size # for example start = 1*1000, end = 2*1000 = 2000, then next start is 2000.
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainloader = data.DataLoader(trainset, batch_size=batchsize, drop_last=False,
                                      num_workers=args.num_workers, sampler=SubsetRandomSampler(indices))
        input_data = inf_train_gen(trainloader) 
        input_pipelines.append(input_data) #a list of generators with wanted indices.


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  TRAINING 
      
    for idx, netD_id in enumerate(net_ids): 
        print("Training discriminator number ", idx, "/", num_discriminators)
        ### stop the process if finished
        if netD_id >= num_discriminators:
            print('ID {} exceeds the num of discriminators, training is finishing'.format(netD_id))
            sys.exit()

        #--- Load one of the Discriminators initialized
        netD = netD_list[idx] 
        optimizerD = optimizerD_list[idx] 
        input_data = input_pipelines[idx] 

        #--- Init a non-private Generator, specific to current D, used in pre-training and discarded. 
        netG = Generator(z_dim = z_dim, hidden_dim= model_dim, n_features_out= n_features).to(device)
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

        #--- Set save paths for current D 
        save_subdir = os.path.join(save_dir, 'netD_%d' % netD_id) 
 
        if os.path.exists(save_subdir): #check if model with this name already exists
            print("netD %d already pre-trained" % netD_id)
        else:
            mkdir(save_subdir)
            
            # --- Train current discriminator
            
            for iter in range(args.pretrain_iterations + 1): #train to a set max

                for p in netD.parameters(): #tell pytorch to keep track of gradients of this D network's parameters
                    p.requires_grad = True

                for iter_d in range(critic_iters):
                    real_data = next(input_data)  
                    real_data = real_data.to(device)

                    # --- Pass real data through disc.
                    netD.zero_grad()
                    D_real_score = netD(real_data)
                    D_real_score = - D_real_score.mean() #the minus is because of how the loss f works. For the critic the loss has to go towards -inf when crit. score is higher.  
                    
                    # --- Pass G(z) fake-example through
                    batchsize = real_data.shape[0] # for G, the batch-size is just the size of the subset a D sees. 
                    noise_D = get_noise(batchsize, z_dim, device) # generate noise
                    fake_data = netG(noise_D)
                    fake_data = fake_data.detach() #tell generator not to update during repeated discr. runs. 
                    
                    D_fake_score = netD(fake_data) #when fake_score goes up loss -> + inf. 
                    D_fake_score = D_fake_score.mean()
                    
                    # --- Calculate D loss with gradient penalty
                    gp = get_gradient_penalty(netD, real_data, fake_data, device, L_gp).to(device) 
                    D_cost = D_fake_score + D_real_score + gp 

                    # --- Update D 
                    D_cost.backward(retain_graph = True)  
                    optimizerD.step()

                #--- Update the temporary G network
                for p in netD.parameters():
                    p.requires_grad = False #to not update the critic during G update, see above where the contrary is done (for p in p.required_grad = True)
                netG.zero_grad()
             
                noise_G = get_noise(batchsize, z_dim, device)
                fake_G = netG(noise_G)
                fake_G.to(device)
                fake_pred_g = netD(fake_G) # D gives score over fake examples 
                G = - fake_pred_g.mean()
                G.backward()  
                optimizerG.step()

            # --- Save pre-trained discriminator
            torch.save(netD.state_dict(), os.path.join(save_subdir, 'netD.pth'))                
           
    print("Pretraining complete")

if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)
