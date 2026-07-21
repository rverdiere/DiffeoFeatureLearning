import torch
import csv
import numpy as np
import os
import numpy as np


SQRT_2PI = torch.sqrt(torch.tensor(2.0 * torch.pi))


def normal_pdf(z):
    return torch.exp(-0.5 * z**2) / SQRT_2PI.to(z.device, z.dtype)


def transform_inputs(x, a=0.1, b=10.0, eps=1e-7):
    x = x.to(dtype=torch.get_default_dtype())

    a = torch.as_tensor(a, dtype=x.dtype, device=x.device)
    b = torch.as_tensor(b, dtype=x.dtype, device=x.device)

    u = (x - a) / (b - a)
    u = u.clamp(eps, 1.0 - eps)

    # Φ^{-1}(u) = √2 erfinv(2u-1)
    z = torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device)) * \
        torch.erfinv(2.0 * u - 1.0)

    return z


def transform_gradients(x, grad_x, a=0.1, b=10.0, eps=1e-7):
    if x.shape != grad_x.shape:
        raise ValueError(
            f"x and grad_x must have the same shape, "
            f"got {x.shape} and {grad_x.shape}."
        )

    z = transform_inputs(x, a=a, b=b, eps=eps)

    a = torch.as_tensor(a, dtype=x.dtype, device=x.device)
    b = torch.as_tensor(b, dtype=x.dtype, device=x.device)

    dx_dz = (b - a) * normal_pdf(z)

    return grad_x * dx_dz


def normalize(x, grad_x):
    return transform_inputs(x), transform_gradients(x, grad_x.squeeze(-1)).unsqueeze(-1)

def normalize_ae(x):
    x-=x.min(axis=0)[0]
    x/=x.max(axis=0)[0]
    return x

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

def load_dataset_ae(fname, dset=None):
    if dset!= None:
        x = torch.from_numpy(np.loadtxt(fname+'train_dset'+str(dset)+'.csv', delimiter=",", dtype=np.float32)).contiguous()
        n,d = x.size()
        grad_x = torch.eye(d,d).repeat((n,1)).reshape((n,d,d))
    else:
        x = torch.from_numpy(np.loadtxt(fname+'test.csv', delimiter=",", dtype=np.float32)).contiguous()
        n,d = x.size()
        grad_x = torch.eye(d,d).repeat((n,1)).reshape((n,d,d))
    
    return (x, grad_x, x)


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
    

    #losses = torch.tensor(losses, dtype=torch.float64)
    #times = torch.tensor(times, dtype=torch.float64)
    print('Saving results in '+fname)
    if id_run != None:
        fname+="run{id_run}_"

    with open(fname+'loss.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(['MSE', 'NRMSE', 'RL1', 'RL2'])
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
