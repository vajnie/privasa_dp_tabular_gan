# DP GAN for synthetic data generation (PRIVASA)

- This work relates to the paper Nieminen, V., Pahikkala, T., & Airola, A. (2023) Empirical evaluation of amplifying privacy by subsampling for GANs to create differentially private synthetic tabular data. Annual Symposium for Computer Science (TKTP'23). CEUR Workshop Proceedings (In Press). This work has been conducted as part of the PRIVASA project funded by Business Finland (grant number 37428/31/2020). The work freely available source code of Chen et al. (see [GitHub - DingfanChen/GS-WGAN: Official implementation of "GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators" (NeurIPS 2020)](https://github.com/DingfanChen/GS-WGAN)) on the GS-WGAN was used as a starting point for this implementation.

This repository's model is easiest to run using the same dataset as in the above publication; the Cardio dataset (S. Ulianova 2019), that is freely available https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset. However, other datasets are easy to plug in also by doing the appropriate preprocessing, creating a Dataset class (pytorch) and adjusting the input and output layer sizes accordingly. 

Package versions used and code confirmed to run with:
- Pytorch (v. 1.10.2) 
- autodp (version 0.2)
- Python 3.6.10,
- cudatoolkit 11.3.1 (note that this depends on your hardware)
- torchvision 0.11.2 


### Files: 

- models.py contains the pytorch model definitions of both D and G and the gradient penalty (GP) (See [Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028). ) calculation.
- config.py, pretrain.sh and train_model.sh are used to set configs and train the model, except for the layer configurations of the NN, which need to be changed in the models.py
- data.py contains the pytorch dataset definition fit for the cardio data.

- For convenience to get the model running easily, there is the cardio_gan_train.csv, which is the cardio dataset preprocessed (minmaxed and one-hotted).


### Data 

Datasets must be preprocessed. Most importantly, categorical variables need to be one-hotted and min-max scaling for different features is required due to the tanH layer in the network. One can change the tanH layer, but this can affect the Lipschitz-1 continuity condition related to the gradient clipping see [Chen et al. 2021](https://arxiv.org/abs/2006.08265), [Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028). 

## Running the model training and evaluation.

First, pretraining the D networks is run with the bash script pretrain.sh, second, the differentially private model is run using the train_model.sh script and last evaluation, is run via the evaluate.sh bash script.

Below there is an overview of the files, what they do and how to use the code in this repo. 



![running_overview](https://github.com/vajnie/privasa_dp_tabular_gan/assets/47028779/4dc78b4d-3aae-4fcd-8420-0076354292ac)
