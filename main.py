import os
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from losses import *
from training_functions import train_featuremap, train_profilefunc, multistart
from preprocess_functions import *
from glow_invnet import glow_invnet
from coupling_functions import fcnn
from plot import plot_profile

# ---------------- CPU setup ----------------
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(1)

# ---------------- Parameters ----------------
benchmark_name = "exp_cos"
fname = 'datasets/'+benchmark_name+"/"
fname_results = 'results/'+benchmark_name+"/"
nb_runs=1
loss_function=poincare_IS
m=1
nb_layer=4
internal_dim_f = 100
act_func=nn.Sigmoid()
batch_size = 80
max_epochs_g = 150
max_epochs_f=300
nb_multistart=4
max_epochs_ms= 30
ms_printfreq=10
learning_rate = 1e-2
dim_aug=2
full_test=True
plot=True
train_f=True
lamb_pen=1
lamb_KL=0.01
sig_KL=0.1

#Loading testing set
(x_test, grad_x_test, y_test) = load_dataset(fname)
n_test, in_dim = x_test.size()
_, out_dim = y_test.size()

param_dict={'Benchmark name': benchmark_name,
            'Number of runs': nb_runs,
            'Loss function': loss_function.__name__,
            'Latent dimension (m)': m,
            'Number of layers coupling flow': nb_layer,
            'Internal dim profile function': internal_dim_f,
            'Batch size': batch_size,
            'Max epochs feature map': max_epochs_g,
            'Max epochs profile function': max_epochs_f,
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
for k in range(nb_runs):
    #--------- Loading dataset-------------------
    (x_train, grad_x_train, y_train) = load_dataset(fname, dset=k)
    n_train, dim = x_train.size()
    _, out_dim = y_train.size()
    print(f"Banchmark function: {benchmark_name}\nTraining samples: {n_train}, Testing samples: {n_test}, Input dimension: {dim}+{dim_aug}, Output dimension: {out_dim}")
    
    #-------- Dimension augmentation ------------
    if dim_aug > 0:
        with torch.no_grad():
            x_train, grad_x_train = inc_dim(x_train, grad_x_train, dim_aug, out_dim)
            x_test, grad_x_test = inc_dim(x_test, grad_x_test, dim_aug, out_dim)
    
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
    #Profile function
    f = nn.Sequential(nn.Linear(m, internal_dim_f), 
                  act_func, 
                  nn.Linear(internal_dim_f, internal_dim_f),
                  act_func,
                  nn.Linear(internal_dim_f, internal_dim_f),
                  act_func,
                  nn.Linear(internal_dim_f, out_dim))

    # ---------------- Training ----------------

    start = time.perf_counter()

    #Feature map training with multistart
    print("Multistart:")
    g = multistart(nb_multistart, g, dataloader, loss_function, max_epoch=max_epochs_ms, print_freq=ms_printfreq, sig=sig_KL, lamb_KL=lamb_KL)
    print("Training feautre map:")
    g,loss = train_featuremap(g, dataloader, loss_function, max_epoch=max_epochs_g, learning_rate=learning_rate, print_freq=30, sig=sig_KL, lamb_KL=lamb_KL)

    #Profile function training
    if train_f==True: 
        print("Training profile function:")
        train_profilefunc(f, g, dataloader, loss_function=MSE, max_epoch=max_epochs_f)
    
    end = time.perf_counter()
    print("Training time:", end - start, "seconds")
    times.append(end-start)
    
    #---------- Testing -------------------------
    if full_test == True:
        #Computing IS loss
        lossIS_test = poincare_IS(g, x_test, grad_x_test)
        #Assembling covariance matrix for FS loss
        cov_vec = torch.cat((torch.ones(g.out_dim), (sig_KL**2)*torch.ones(g.in_dim-g.out_dim)))
        cov_inv_mat = torch.diag(1/cov_vec)
        #Computing FS loss
        lossFS_test = poincare_FS(g, x_test, grad_x_test, cov_inv_mat, lamb=lamb_KL)
        print(f"POINCARE ERROS\n\tIS loss: {lossIS_test:>6f} FS loss: {lossFS_test:>6f}")

    # ---------------- Evaluation ----------------
    y_pred  = f(g(x_test))
    mse, nrmse, rl1, rl2 = compute_errors(y_test, y_pred)
    losses.append([mse, nrmse, rl1, rl2])

    print(f"TEST ERRORS\n\tMSE: {mse:>6f} NRMSE: {nrmse:>6f} RL1: {rl1:>6f} RL2: {rl2:>6f}")
    
    #------------- Plotting ----------------------
    if plot == True:
        plot_profile(f, g, x_test, y_test) 

#Printing results
if len(losses)>1:
    print_results(losses,times)
#Saving results
save_results(losses, times, param_dict, fname_results)
