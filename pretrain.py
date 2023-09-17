import os, sys
import numpy as np
import random
import torch
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler

from config import *
from models import *
from utils import *

sys.path.insert(0, './data')
from cardio_data import *

### my_pretrain.py 
##########################################################################################################################################################
#Here we pretrain discriminators. How many depends on @n_discriminators from pretrain.sh. 
#Everything run here is WITHOUT differential privacy. According to theory presented in Chen et. al GS-WGAN
#we can pretrain discriminators with non-private generators as we only aim to release the final generator,
#that will be trained with differential privacy applied.

# This model is based on the work and code of Chen, Orekondy and Fritz https://github.com/DingfanChen/GS-WGAN
# and the paper of the aforementioned authors GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators, 2020-12-06, NeurIPS
# The code has been modified to work with a WGAN-GP implementation by Valtteri Nieminen as part of his masters thesis. Accommodations to be able to train
# with tabular data and to get better results are done as well as simplifying and modifying the code for this purpose. The privacy calculation remains
# mostly untouched. 

### Fix noise for visualization and visualizations are deleted from this in comparison to original
#(See GSWGAN pretrain.py lines 49-58 and 206-218 deleted from here as unused, visited 3.11.2021)
#https://github.com/DingfanChen/GS-WGAN/blob/main/source/pretrain.py
#--------------------------------------------------------------------------------------------------------------------------------------------------------#


##########################################################
### main
##########################################################
#Parse arguments and setup.
#---------------------------------------------------------
def main(args):
    
    n_features_out = args.n_features_out
    num_discriminators = args.num_discriminators
    z_dim = args.z_dim
    model_dim = args.model_dim
    batchsize = args.batchsize
    L_gp = args.L_gp
    critic_iters = args.critic_iters
    save_dir = args.save_dir
    net_ids = args.net_ids

    print("Number of discriminators:", num_discriminators)

    ### CUDA
    #------------------------------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ### Random seed
    #------------------------------
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    ############################################################
    ### Set up discriminator models
    ############################################################
    netD_list = [] #list of all discriminators trained
    for _ in range(len(net_ids)): 
        netD = Critic(model_dim, n_features_out)
        netD_list.append(netD)
    netD_list = [netD.to(device) for netD in netD_list]

    ### Set up optimizers for discr.
    #--------------------------------
    optimizerD_list = []
    for i in range(len(net_ids)):
        netD = netD_list[i]
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerD_list.append(optimizerD)


    ### DATA
    #-------------------------------
    #DataLoader
    #----------
    #@trainset_size is the size individual train set for one discriminator. 
    cardio_train_df = pd.read_csv("/data/gan_training_data.csv") 
    # [1] is just the labels for convenience. 

    trainset = Dataset(cardio_train_df, device)

    #Create indices files - 
    # (indices for individual discriminators) 
    #----------------------------------------
    if os.path.exists(os.path.join(save_dir, 'indices.npy')):
        print('load indices from disk')
        indices_full = np.load(os.path.join(save_dir, 'indices.npy'), allow_pickle=True)
    else:
        print('create indices file')
        indices_full = np.arange(len(trainset))
        np.random.shuffle(indices_full)
        indices_full.dump(os.path.join(save_dir, 'indices.npy'))
    trainset_size = int(len(trainset) / num_discriminators)
    print('Size of the dataset: ', trainset_size)

    # Input pipelines for individual discriminators
    # - make a generator for each that indexes into training data.
    ###############################################################

    
    input_pipelines = []
    for i in net_ids:
        start = i * trainset_size # for example start = 1*1000, end = 2*1000 = 2000, then next start is 2000.
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainloader = data.DataLoader(trainset, batch_size=batchsize, drop_last=False,
                                      num_workers=args.num_workers, sampler=SubsetRandomSampler(indices))
        input_data = inf_train_gen(trainloader) 
        input_pipelines.append(input_data) #a list of generators with wanted indices.


    ###########################################
    ### Training Loop
    # "for each discriminator to be pretrained"
    # each discriminator gets its own 
    # non-private generator. 
    ###########################################

    
    for idx, netD_id in enumerate(net_ids):
        print("Training discriminator number ", idx, "/", num_discriminators)
        ### stop the process if finished
        if netD_id >= num_discriminators:
            print('ID {} exceeds the num of discriminators, training is finishing'.format(netD_id))
            sys.exit()

        #Load a Discriminator
        #####################
        netD = netD_list[idx]
        optimizerD = optimizerD_list[idx]
        input_data = input_pipelines[idx]

        #Load a Generator 
        #####################
        netG = Generator(z_dim = z_dim, hidden_dim= model_dim, n_features_out=n_features_out).to(device)
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

        ### Set save paths, check if already trained.
        #############################################
        save_subdir = os.path.join(save_dir, 'netD_%d' % netD_id)

        if os.path.exists(save_subdir):
            print("netD %d already pre-trained" % netD_id)
        else:
            mkdir(save_subdir)
            
            ### Train current discriminator and generator
            ################################################
            for iter in range(args.pretrain_iterations + 1):

                #########################
                ### Update D network
                #########################
                for p in netD.parameters():
                    p.requires_grad = True

                #Training iteration(s) for a discriminator
                #-------------------------------------------------------------------------------------------------------------------------
                #@input_data has both X and y in them and we needo only x here (x contains target var too as we will synthesize that too). 
                #@input_data is a tuple with input_data[0] being all variables (including target) and [1] having only target.
                        
                for iter_d in range(critic_iters):
                    real_data, real_y = next(input_data)
                    real_data = real_data.to(device)

                    # Pass real data through disc.
                    #---------------------------------
                    netD.zero_grad()
                    D_real_score = netD(real_data)
                    D_real_score = - D_real_score.mean() #the minus is because of how the loss f works. For the critic the loss has to go towards -inf when crit. score is higher.  
                    # Pass fake data through
                    #---------------------------------
                    batchsize = real_data.shape[0]
                    noise_D = get_noise(batchsize, z_dim, device) # generate noise
                    fake_data = netG(noise_D)
                    fake_data = fake_data.detach() #This is called so that the generator doesnt update during discr. runs. 
                    
                    D_fake_score = netD(fake_data) #no - here so when fake_score goes up loss -> + inf. 
                    D_fake_score = D_fake_score.mean()
                    
                    #Calculate D cost with gradient penalty
                    #--------------------------------------
                    gp = get_gradient_penalty(netD, real_data, fake_data, device, L_gp).to(device) 
                    D_cost = D_fake_score + D_real_score + gp

                    ### update
                    D_cost.backward(retain_graph = True) #this is due to how the computational graph is built in our gs-wgan solution - generator and discr. are part of the same graph. 
                    #Wasserstein_D = -D_real - D_fake #this is for visualization., Does not work at the moment  
                    optimizerD.step()

                ############################
                # Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False #To not update the critic, sama kuin detach
                netG.zero_grad()
             

                noise_G = get_noise(batchsize, z_dim, device)
                fake_G = netG(noise_G)
                fake_G.to(device)
                fake_pred_g = netD(fake_G) 
                G = - fake_pred_g.mean()

                ### update G
                #-----------
                G.backward() #DP hook is fired here
                G_cost = G
                optimizerG.step()
                torch.save(netD.state_dict(), os.path.join(save_subdir, 'netD.pth'))                
           
    print("Pretraining complete")

if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)
