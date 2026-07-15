import torch
from skopt.sampler import Lhs
import csv
import numpy as np
import os

def uniToNormal(X, a=0.1, b=10.0):
    return torch.distributions.Normal(0, 1).icdf((X-a)/(b-a))

def normalToUni(X, a=0.1, b=10.0):
    return (b-a)*torch.distributions.Normal(0, 1, validate_args=False).cdf(X)+a

def LHS_sampling(n_samples, dim, l_bound, u_bound):
    lhs = Lhs(lhs_type='classic', criterion='maximin', iterations=1000) 
    dim  = [(float(l_bound), float(u_bound)) for k in range(dim)]
    sample = lhs.generate(dim, n_samples)
    return torch.tensor(sample, requires_grad=True)

def load_dataset(fname, dset=None):
    if dset!= None:
        x = torch.from_numpy(np.loadtxt(fname+'train_X_dset'+str(dset)+'.csv', delimiter=",", dtype=np.float32)).contiguous()
        grad_x = torch.from_numpy(np.loadtxt(fname+'train_guX_dset'+str(dset)+'.csv', delimiter=",", dtype=np.float32)).contiguous()
        y = torch.from_numpy(np.loadtxt(fname+'train_uX_dset'+str(dset)+'.csv', delimiter=",", dtype=np.float32)).contiguous()
    else:
        x = torch.from_numpy(np.loadtxt(fname+'test_X.csv', delimiter=",", dtype=np.float32)).contiguous()
        grad_x = torch.from_numpy(np.loadtxt(fname+'test_guX.csv', delimiter=",", dtype=np.float32)).contiguous()
        y = torch.from_numpy(np.loadtxt(fname+'test_uX.csv', delimiter=",", dtype=np.float32)).contiguous()
    
    if y.dim()==1:
        y= y.unsqueeze(-1)
    
    _, out_dim = y.size()
    n, dim = x.size()
    grad_x = grad_x.reshape((n, out_dim, dim))
    grad_x = grad_x.transpose(-1,-2)

    return (x, grad_x, y)

def inc_dim(x, grad_x, dim_aug, out_dim=1):
    with torch.no_grad():
        n, dim = x.size()
        x_aug = torch.randn((n, dim_aug))
        z = torch.zeros((n, dim_aug, out_dim))
        x = torch.cat((x, x_aug), dim=1)
        grad_x = torch.cat((grad_x, z), dim=1)
    return x, grad_x

def save_results(losses, times, param_dict, fname, id_run=None):
    #Creating directory if it doesn't exist already
    try:
        os.mkdir(fname)
        print(f"Directory '{fname}' created successfully.")
    except FileExistsError:
        print(f"Directory '{fname}' already exists.")
    

    losses = torch.tensor(losses, dtype=torch.float64)
    times = torch.tensor(times, dtype=torch.float64)
    print('Saving results in '+fname)
    if id_run != None:
        fname+="run{id_run}_"

    with open(fname+'loss.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(losses)
    with open(fname+'times.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(losses)
    with open(fname+'parameters.csv', 'w', newline='') as myfile:
        writer = csv.DictWriter(myfile, fieldnames=param_dict.keys())
        writer.writeheader()
        writer.writerow(param_dict)

def print_results(losses, times):
    res = torch.tensor(losses, dtype=torch.float64).mean(dim=0)
    res_std = torch.tensor(losses, dtype=torch.float64).std(dim=0)
    time = torch.tensor(times).mean()
    time_std = torch.tensor(times).std()
    print(f"MSE mean: {res[0]} std: {res_std[0]} \n NRMSE mean: {res[1]} std: {res_std[1]} \n RL1 mean: {res[2]} std: {res_std[2]}\n RL2 mean: {res[3]} std: {res_std[3]}")
    print(f"Training time mean: {time} std: {time_std}")
