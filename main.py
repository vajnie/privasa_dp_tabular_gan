import os
import sys
import numpy as np
import random
import copy
import torch
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler

#--Import custom functions --
from my_config import * #config parsing functions
from my_models import * #model creation and specifications
from my_utils import *  

#-- Import data-loading functions --
from cardio_data import * #import dataset class


#------------------------------------------------------------------------------------------------------------------
#main.py: training of the differentially private generator. Settings can be changed in config or as args.
# @seed, number of discriminators and training iterations for runs were specified in the bash script train_model.sh
#------------------------------------------------------------------------------------------------------------------
# PACKAGE VERSIONS USED:
# pytorch 1.4.0
# torchvision 0.2.1
# cudatoolkit 10.0.130

# NOTES: -----------------------------------------------------------------------------------------#

# This code is based partially on the work of Chen et. al GS-WGAN see
# https://github.com/DingfanChen/GS-WGAN/
#-------------------------------------------------------------------------------------------------#

# Privacy settings 
#---------------------------
CLIP_BOUND = 1.
SENSITIVITY = 2.


# Hook functions 
#---------------------------
def master_hook_adder(module, grad_input, grad_output):
    ''' Global dynamic function, that is used to change the hook that is used'''
    global dynamic_hook_function
    return dynamic_hook_function(module, grad_input, grad_output)


def dummy_hook(module, grad_input, grad_output):    
    ''' One of the hooks used with global dynamic function that does nothing.
    The reference of the dynamic function is updated to point to this function 
    when the discriminator is trained as no gradients are modified during discr. training'''
    pass

    # Changed to pytorch full_backward_hook @26.09.2022
def dp_hook(module, grad_input, grad_output):
    
    ''' The differentially private hook. Sanitize the upstream gradients on the 
    boundary between discriminator and generator during backpropagation by clipping and adding noise, changed to support new pytorch versions 26.09.2022
    by changing the dp_hook to full_backward hook and making fixes. 
    May be important since in older versions of pytorch, (see https://github.com/pytorch/pytorch/issues/598)
    there is a bug with the linear layer where the order of the returned gradients in grad_input is different from other layers
    We take the grad input coming to the node this is attached to, we modify the input and then the modified input will be differentiated '''
    
    # - Take the upstream gradient -
    # "gradient of propagated loss score with respect to input of this layer and output of previous."
    #-------------------------------------------------------------------------
    grad_wrt_input = grad_input[0] #grad input is in terms of backpropagation, so this is the last layer of the G network (viewed left to right)
    grad_input_size = grad_wrt_input.size()

    
    # Uncomment prints to sanity check. 
    # ----------------------------------------------
    #print('***'*5+'  Grad in, unmodified '+'***'*5)
    #print(grad_wrt_input)

     #check that we have the right size gradient from tuple. 
    # see: https://github.com/pytorch/pytorch/issues/598
    
    ### calculate clip bound, norm of the grad
    #-----------------------------------------
    clip_bound_ = CLIP_BOUND / batchsize #this is to account for GP
    grad_input_norm = torch.norm(grad_wrt_input, p = 2, dim = 1) #l2 norm, by row, returns list length of batch
    
    gradient_norms.append([x.item() for x in grad_input_norm])    
    #print('***'*5+'  Grad input norms, unmodified '+'***'*5)
    #print(grad_input_norm)
    ### clip
    #-----------------------------------------
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_input = clip_coef * grad_wrt_input #clipped grad

    ### Add noise
    #----------------------------------------
    noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_input) #create gaussian noise
    grad_wrt_input = grad_wrt_input + noise
  
    ### Revert to the original form of the tuple and return
    #------------------------------------------------------
    #print('***'*5+'  Modified gradient '+'***'*5)
    #print(grad_wrt_input)
    grad_wrt_input = ((grad_wrt_input), ) 
    return(grad_wrt_input)
    

##########################################################
### Main
##########################################################
def main(args):
    ### config
    global noise_multiplier
    global batchsize
    num_discriminators = args.num_discriminators
    noise_multiplier = args.noise_multiplier
    z_dim = args.z_dim
    model_dim = args.model_dim
    batchsize = args.batchsize
    L_gp = args.L_gp
    critic_iters = args.critic_iters
    load_dir = args.load_dir
    save_dir = args.save_dir
    num_gpus = args.num_gpus
    n_features_out = args.n_features_out
    split = args.splits
    

    global gradient_norms
    gradient_norms = [] #for recording grad norms
    
    ### CUDA
    #----------------------------------
    use_cuda = torch.cuda.is_available()
    
    devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(num_gpus)]
    device0 = devices[0]
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    device = device0
    ### Random seed
    seed = random.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    ########################################
    ### Set up models
    ########################################

    # Init generator and generator optimizer.
    #----------------------------------------
    netG = Generator(z_dim = z_dim, hidden_dim= model_dim, n_features_out=n_features_out).to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
   

    
    ### Initialize "empty" discriminators 
    #------------------------------------
    netD_list = []
    for i in range(num_discriminators):
        netD = Critic(model_dim, n_features_out)
        netD_list.append(netD)

    ### Load the weights of the pre-trained discriminators 
    #-----------------------------------------------------
    if load_dir is not None:
        for netD_id in range(num_discriminators):
            #print('Load NetD ', str(netD_id))
            network_path = os.path.join(load_dir, 'netD_%d' % netD_id, 'netD.pth')
            netD = netD_list[netD_id]
            netD.load_state_dict(torch.load(network_path))

    netG = netG.to(device0)
    for netD_id, netD in enumerate(netD_list):
        device = devices[get_device_id(netD_id, num_discriminators, num_gpus)]
        netD.to(device)

    ### Set up optimizers for discriminators
    #------------------------------------------------------
    optimizerD_list = []
    for i in range(num_discriminators):
        netD = netD_list[i]
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerD_list.append(optimizerD)
    

    #------------------------------------------------------
    ### Data loading
    #------------------------------------------------------
    
    #DataLoader
    #----------
    #@trainset_size is the size individual train set for one discriminator. 
    cardio_train_df = pd.read_csv("/home/vajnie/thesis/data/gan_training_data.csv") 
    trainset = Dataset(cardio_train_df, device)

    #Load indices of the data that each of the pre-tr. discr. relate to. 
    #-------------------------------------------------------------------
    if load_dir is not None:
        assert os.path.exists(os.path.join(load_dir, 'indices.npy'))
        print('load indices from disk')
        indices_full = np.load(os.path.join(load_dir, 'indices.npy'), allow_pickle=True)
    else:
        print('creat indices file')
        indices_full = np.arange(len(trainset))
        np.random.shuffle(indices_full)
        indices_full.dump(os.path.join(save_dir, 'indices.npy'))
    trainset_size = int(len(trainset) / num_discriminators)
    print('Size of the dataset: ', trainset_size)

    #Input pipelines for the discriminators as in pretraining
    #--------------------------------------------------------
    input_pipelines = []
    for i in range(num_discriminators):
        start = i * trainset_size
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainloader = data.DataLoader(trainset, batch_size=args.batchsize, drop_last=False,
                                      num_workers=args.num_workers, sampler=SubsetRandomSampler(indices))
        input_data = inf_train_gen(trainloader)
        input_pipelines.append(input_data)

    ### Register hook
    #--------------------------------------------------------
    global dynamic_hook_function
    for netD in netD_list:
    #Add the hook to the critics first (in terms of forward pass) layer.
        netD.crit[0][0].register_backward_hook(master_hook_adder)

 
    ################################################
    ### Update D network
    ################################################

    for p in netD.parameters():
            p.requires_grad = True


    for iter in range(args.iterations + 1):
        if(iter % 1000 == 1):
            print("Passed iteration ", iter, "out of", args.iterations)    
        
        #Here we pick and load a random discriminator from the pre-trained discriminators.
        #---------------------------------------------------------------------------------
        #This is what gives us the "amplification by subsampling" benefit in terms of DP guarantees.
        netD_id = np.random.randint(num_discriminators, size=1)[0]
        device = devices[get_device_id(netD_id, num_discriminators, num_gpus)]
        netD = netD_list[netD_id]
        optimizerD = optimizerD_list[netD_id]
        input_data = input_pipelines[netD_id]

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
            dynamic_hook_function = dummy_hook #this does not add dp or anything else since we are training discriminators that dont need it. 
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


        del real_data, real_y, noise_D, fake_data, D_real_score, D_fake_score, gp
        torch.cuda.empty_cache()

        #####################################################################
        # Update G network
        #####################################################################

        ### Sanitize the gradients passed to the Generator
        dynamic_hook_function = dp_hook
        
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()


        ### Train with sanitized discriminator output
        # the hook is attached to the "backward pass last layer" for all discriminators
        # the gradient that comes to G is already sanitized.
        #-------------------------------------------------------------------------------
        

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
    
        torch.save(netD.state_dict(), os.path.join(save_dir, 'netD.pth'))
        del fake_G,  fake_pred_g, noise_G, G_cost, D_cost
        torch.cuda.empty_cache()

    
        ### Save an intermediate model
        # save_dir path is for example ~/projects/gradu/gan_models/{name}
        # so we need to make a dir for save_dir + intermediate
        #----------------------------------------------------------------

        intermediate_model_path = save_dir + "/intermediate"
        mkdir(intermediate_model_path)
        if((iter % 1000 == 1) and (iter > 999)):
            print("Saving intermediate model ", iter, "out of", args.iterations, "iterations")
            torch.save(netG.state_dict(), (intermediate_model_path + "/" + 'iter' + str(iter) + '_netG.pth'))
           

            #give model a name and tunniste, where does model name come from?    

    ### save model
    torch.save(netG.state_dict(), os.path.join(save_dir, 'netG.pth'))
   
print("Training complete!")

if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)