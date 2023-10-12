import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F



########################################
###        GRADIENT PENALTY          ### 
########################################

def get_gradient_penalty(crit, real, fake, device, L_gp):
    """ Get gradient of the critic's scores with respect to real and fake observations (this is the interpolation), 
    but just a getter, the gradient penalty is calculated in gradient_penalty()    
    Parameters:
        crit: the critic model
        real: a batch of real data
        fake: a batch of fake data
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """

    real = real.to(device)
    fake = fake.to(device)
    
    #Create epsilon for interpolating.
    #-----------------------------------
    epsilon = torch.rand(len(real), 1, requires_grad=True).to(device) #this is not dp epsilon its gradient penalty. Size: (batch_size, 1)
        
    # Mix inputs together
    #---------------------
    interpolates = (real * epsilon) + (fake * (1 - epsilon))
    interpolates = interpolates.to(device)
    
    # Score
    #-------
    crit_interpolates = crit.forward(interpolates)
    
    # Take the gradient of the scores with respect to inputs
    gradient = torch.autograd.grad(
        inputs=interpolates,
        outputs=crit_interpolates, # because autograd gets the crit_interpolates here it is able to link them with the discr. graph 
        grad_outputs=torch.ones(crit_interpolates.size()).to(device),  #output a vector of gradients for each input. 
        create_graph=True,
        retain_graph=True,
        only_inputs = True
    )[0]

    # Calculate the magnitude of every column
    gradient_norm = gradient.norm(2, dim=1)
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = ((gradient_norm -1)**2).mean() * L_gp
    return(penalty)



    

########################################
###           GENERATOR              ### 
########################################


class Generator(nn.Module):
    '''
    This is the custom generator used for tabular data. 
    '''
    def __init__(self, z_dim, hidden_dim, n_features_out):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        #init network 
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, n_features_out, final_layer=True),
        )
        
        '''
        Make a generator block with batchnorm, ReLU except in last layer where we have TanH 
        Parameters:
            in_features: the size of input - size of last layer output
            out_features: output_size of this layer
            final_layer: true if it is the final layer and false otherwise 
                      final layer does not have batchnorm and uses TanH
        '''
    def make_gen_block(self, in_features, out_features, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Tanh(),
            )
    def forward(self, noise):
        '''
        returns generated data
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise) 

### Make noise vectors in batches ### 
def get_noise(n_samples, z_dim, device):
    '''
    Create batches of z-noise vectors
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate (bath size) a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
        '''
    return torch.randn(n_samples, z_dim, device = device)   

########################################
###         Discriminator            ### 
########################################

class Critic(nn.Module):
    '''
    Values:
        hidden_dim: inner_dim in network, here the first is == len(len_generated_features.columns) (amount of cols) 
    '''
    def __init__(self, hidden_dim, in_features_len):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(in_features_len, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, in_features, out_features, final_layer=False):
        '''
        Make one critic block.
        Batchnorm, LeakyRelu with 0.2 as default.  
        final_layer = True will results in block without batchnorm.
        otherwise arguments same as in generator. 
        @return: a scalar that models the Earth-movers distance of the distr of Dgen vs Dreal. 
                    '''
        if not final_layer:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
            )

    def forward(self, observation):
        crit_pred = self.crit(observation)
        return crit_pred.view(len(crit_pred), -1)

