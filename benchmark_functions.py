import torch
from torch.func import grad
import numpy as np
from numpy import pi
from preprocess_functions import LHS_sampling, uniToNormal, normalToUni

class benchmarkFunc:
    def __init__(self, dim, domain_low=-1, domain_high=1):
        self.name="func"
        self.dim=dim
        self.domain_low=float(domain_low)
        self.domain_high=float(domain_high)
        self.domain = "["+str(self.domain_low)+";"+str(self.domain_high)+"]"
    
    def samples(self, n_samples, LHS=False):
        if LHS:
            x = LHS_sampling(n_samples, self.dim, self.domain_low, self.domain_high)
            return uniToNormal(x, self.domain_low, self.domain_high)
        else:
            return torch.randn(n_samples,self.dim)

class exp_cos(benchmarkFunc):
    def __init__(self, dim=8, domain_low=-pi/2, domain_high=pi/2):
        super().__init__(dim, domain_low, domain_high)
        self.name="exp_cos"

    def func(self, X):
        X = normalToUni(X, self.domain_low, self.domain_high)
        return torch.exp(torch.mean(torch.sin(X)*torch.exp(torch.cos(X))))

    def grad(self, X):
        return grad(self.func)(X)

class cos_exp(benchmarkFunc):
    def __init__(self, dim=8, domain_low=-1, domain_high=1):
        super().__init__(dim, domain_low, domain_high)
        self.name="cos_exp"

    def func(self, X):
        X = normalToUni(X, self.domain_low, self.domain_high)
        return torch.cos(X[-1]*torch.exp(torch.sum(torch.sigmoid(X[:-1]))))

    def grad(self, X):
        return grad(self.func)(X)


class sin_norm(benchmarkFunc):
    def __init__(self, dim=8, domain_low=-1, domain_high=1):
        super().__init__(dim, domain_low, domain_high)
        self.name="sin_norm"
    
    def func(self, X, norm_const=1):
        X = normalToUni(X, self.domain_low, self.domain_high)[:self.dim]
        return torch.sin(1/norm_const*torch.sum(X**2))

    def grad(self, X):
        return grad(self.func)(X)
