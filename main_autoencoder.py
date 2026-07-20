import os
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from src.losses import *
from src.training_functions import train_featuremap, train_profilefunc, multistart
from src.preprocess_functions import *
from src.glow_invnet import glow_invnet
from src.coupling_functions import fcnn
from src.plot import plot_profile

# ---------------- CPU setup ----------------
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(1)

# ---------------- Parameters ----------------
benchmark_name = "q16_100"
standard_scale=True
fname = 'datasets/autoencoders/'+benchmark_name+"/"
fname_results = 'results/'+benchmark_name+"/"
nb_runs=2
loss_function=poincare_FS_ae
m=5
nb_layer=4
act_func=nn.Sigmoid()
batch_size = 10
max_epochs_g = 150
nb_multistart=4
max_epochs_ms= 30
ms_printfreq=10
learning_rate = 1e-2
dim_aug=0
full_test=True
lamb_pen=1
lamb_KL=0.01
sig_KL=0.1

#Loading testing set
(x_test, grad_x_test, y_test) = load_dataset_ae(fname)
n_test, in_dim = x_test.size()
out_dim = in_dim

param_dict={'Benchmark name': benchmark_name,
            'Number of runs': nb_runs,
            'Loss function': loss_function.__name__,
            'Latent dimension (m)': m,
            'Number of layers coupling flow': nb_layer,
            'Batch size': batch_size,
            'Max epochs feature map': max_epochs_g,
            'Number of multistart feature map': nb_multistart,
            'Max epochs multistart': max_epochs_ms,
            'Learning rate': learning_rate,
            'Input dimension': in_dim,
            'Output dimension': out_dim,
            'Augmented dimension': dim_aug,
            'Lambda KL poincare FS loss': lamb_KL,
            'Sigma covariance matrix poincare FS loss': sig_KL
            }

losses=[]
times = []
if standard_scale==True:
    with torch.no_grad():
        x_test = normalize_ae(x_test)
if dim_aug > 0:
    with torch.no_grad():
        x_test, grad_x_test = inc_dim(x_test, grad_x_test, dim_aug, out_dim)

for k in range(nb_runs):
    #--------- Loading dataset-------------------
    (x_train, grad_x_train, y_train) = load_dataset_ae(fname, dset=k)
    n_train, dim = x_train.size()
    print(f"Banchmark: {benchmark_name}\nTraining samples: {n_train}, Testing samples: {n_test}, Input dimension: {dim}+{dim_aug}, Output dimension: {dim}")
    
    #-------- Normalization----------------------
    if standard_scale==True:
        with torch.no_grad():
            x_train = normalize_ae(x_train)

    #-------- Dimension augmentation ------------
    if dim_aug > 0:
        with torch.no_grad():
            x_train, grad_x_train = inc_dim(x_train, grad_x_train, dim_aug, out_dim)
    
    #-------- Dataset preparation ---------------
    trainset = TensorDataset(x_train, grad_x_train, y_train)
    dataloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        persistent_workers=True
    )
    
    # ---------------- Model ----------------
    
    #Feature map
    g = glow_invnet(dim+dim_aug, m, nb_layer, fcnn, internal_dim_st=int((dim+dim_aug)/2), nb_layer_st=1)

    # ---------------- Training ----------------

    start = time.perf_counter()

    #Feature map training with multistart
    print("Multistart:")
    g = multistart(nb_multistart, g, dataloader, loss_function, max_epoch=max_epochs_ms, print_freq=ms_printfreq, sig=sig_KL, lamb_KL=lamb_KL)
    print("Training feautre map:")
    g,loss = train_featuremap(g, dataloader, loss_function, max_epoch=max_epochs_g, learning_rate=learning_rate, print_freq=30, sig=sig_KL, lamb_KL=lamb_KL)

    
    end = time.perf_counter()
    print("Training time:", end - start, "seconds")
    times.append(end-start)
    
    #---------- Testing -------------------------
    if full_test == True:
        #Assembling covariance matrix for FS loss
        cov_vec = torch.cat((torch.ones(g.out_dim), (sig_KL**2)*torch.ones(g.in_dim-g.out_dim)))
        cov_inv_mat = torch.diag(1/cov_vec)
        #Computing FS loss
        lossFS_test = poincare_FS_ae(g, x_test, cov_inv_mat, lamb=lamb_KL)
        print(f"POINCARE ERROS\n\tFS loss: {lossFS_test:>6f}")

    # ---------------- Evaluation ----------------
    _, x_pred  = g.autoencoder(x_test)
    mse, nrmse, rl1, rl2 = compute_errors(x_test, x_pred)
    losses.append([mse, nrmse, rl1, rl2])

    print(f"TEST ERRORS\n\tMSE: {mse:>6f} NRMSE: {nrmse:>6f} RL1: {rl1:>6f} RL2: {rl2:>6f}")
    
#Printing results
if len(losses)>1:
    print_results(losses,times)
#Saving results
save_results(losses, times, param_dict, fname_results)
