import torch
import os
import numpy as np
from torch import vmap
from benchmark_functions import sin_norm, exp_cos, cos_exp

def generate_dset(f, n, LHS=False):
    x = f.samples(n, LHS)
    grad_x = vmap(f.grad)(x)
    y = vmap(f.func)(x)
    return x, grad_x, y.reshape((n, 1))

def tensorToCsv(x, path):
    x = np.asarray(x.tolist())
    np.savetxt(path, x, delimiter=",")

if __name__ == "__main__":
    #Datasets parameters
    n_train = 500
    n_test = 1000
    LHS = True
    dim_in = 20
    nb_dset=10

    #bench_f = cos_exp(dim_in, domain_low=-1, domain_high=1)
    #bench_f = exp_cos(dim_in, domain_low=-1, domain_high=1)
    bench_f = sin_norm(dim_in, domain_low=0, domain_high=1)
    path = f'{bench_f.name}/'

    #Creating directory if it doesn't exist already
    try:
        os.mkdir(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    
    # Generating training sets
    for k in range(nb_dset):
        print("Generating dataset "+str(k))
        
        # Generate training set k
        x, grad_x, y = generate_dset(bench_f, n_train, LHS)
        
        #Save dataset k
        tensorToCsv(x, path+'train_X_dset'+str(k)+'.csv')
        tensorToCsv(grad_x, path+'train_guX_dset'+str(k)+'.csv')
        tensorToCsv(y, path+'train_uX_dset'+str(k)+'.csv')

    #Generating testing set
    x, grad_x, y = generate_dset(bench_f, n_test, False)

    #Saving testing set as CSV
    tensorToCsv(x, path+'test_X.csv')
    tensorToCsv(grad_x, path+'test_guX.csv')
    tensorToCsv(y, path+'test_uX.csv')
