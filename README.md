# DP GAN for synthetic data generation
- This work relates to the paper Nieminen, V., Pahikkala, T., & Airola, A. (2023) Empirical evaluation of amplifying privacy by subsampling for GANs to create differentially private synthetic tabular data. Annual Symposium for Computer Science (TKTP'23). CEUR Workshop Proceedings (In Press). This work has been conducted as part of the PRIVASA project funded by Business Finland (grant number 37428/31/2020)

This repository's model is easiest to run using the same dataset as in the above publication; the Cardio dataset (S. Ulianova 2019), that is freely available https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset. However, other datasets are easy to plug in also by doing the appropriate preprocessing, creating a Dataset class (pytorch) and adjusting the input and output layer sizes accordingly. 



