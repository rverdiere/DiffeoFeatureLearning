import torch
import torch.nn as nn

def fcnn(input_dim, output_dim, internal_dim, nb_layer=1, act_func=nn.Tanh()):
    ret_nn = nn.Sequential(nn.Linear(input_dim, internal_dim), act_func)
    for k in range(nb_layer-1):
        ret_nn.append(nn.Linear(internal_dim, internal_dim))
        ret_nn.append(act_func)
    ret_nn.append(nn.Linear(internal_dim, output_dim))
    ret_nn.append(act_func)
    return ret_nn


