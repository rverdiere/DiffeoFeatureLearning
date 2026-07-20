import torch
import torch.nn as nn
import torch.linalg as alg
from torch import vmap
from torch.func import jacrev
import numpy as np

class glow_block(nn.Module):
    def __init__(self, dim, s, t):
        super().__init__()

        self.dim = dim
        self.dim_cf = dim // 2

        self.s = s
        self.t = t
        self.theta = nn.Parameter(torch.tensor(np.pi/2))
        #self.theta = torch.tensor(np.pi/2)
    def get_W(self):

        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
    
        return torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c])
        ])

    def reset_parameters(self):
        for m in (self.s, self.t):
            for subm in m:
                if isinstance(subm, nn.Linear):
                    subm.reset_parameters()

    def permute(self, x, rev=False):
        W = self.get_W()

        if x.ndim == 1:
            x_r = x.reshape(2, self.dim_cf)
            if rev:
                return (alg.inv(W) @ x_r).reshape(self.dim)

            return (W @ x_r).reshape(self.dim)

        elif x.ndim == 2:
            batch = x.shape[0]
            x_r = x.reshape(batch,2,self.dim_cf,)

            W_use = (alg.inv(W) if rev else W)

            out = torch.einsum("ij,bjk->bik", W_use,x_r,)

            return out.reshape(batch, self.dim, )

        raise ValueError("Expected 1D or 2D tensor.")

    def log_jacobian(self, x):

        if x.ndim == 1:
            u, _ = torch.split(x, self.dim_cf,)
            return torch.sum(self.s(u))

        elif x.ndim == 2:

            u, _ = torch.split(x,self.dim_cf, dim=1,)
            return torch.sum(self.s(u),dim=1,)

        raise ValueError("Expected 1D or 2D tensor.")

    def forward(self, x, rev=False):

        if x.ndim == 1:
            if rev:
                x = self.permute(x, rev=True)
                u, v = torch.split(x, self.dim_cf,dim=0)
                s = self.s(u)
                t = self.t(u)

                v = (v - t) * torch.exp(-s)

                return torch.cat((u, v), dim=0, )

            else:
                u, v = torch.split(x, self.dim_cf, dim=0)
                s = self.s(u)
                t = self.t(u)
                v = v * torch.exp(s) + t

                out = torch.cat((u, v), dim=0,)
                return self.permute(out)

        elif x.ndim == 2:
            if rev:
                x = self.permute(x, rev=True)
                u, v = torch.split(x, self.dim_cf, dim=1)
                s = self.s(u)
                t = self.t(u)

                v = (v - t) * torch.exp(-s)
                return torch.cat((u, v), dim=1, )

            else:
                u, v = torch.split(x,self.dim_cf,dim=1)

                s = self.s(u)
                t = self.t(u)
                v = v * torch.exp(s) + t

                out = torch.cat((u, v),dim=1,)

                return self.permute(out)

        raise ValueError("Expected 1D or 2D tensor.")

class glow_invnet(nn.Module):
    
    def __init__(self, input_dim, output_dim, nb_layer, nn_constructor, internal_dim_st=10, nb_layer_st=3):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.dim_cf = int(self.in_dim/2)
        self.nb_layer = nb_layer
        self.internal_dim_st = internal_dim_st
        self.nb_layer_st = nb_layer_st
        self.seq = nn.Sequential()
        for i in range(self.nb_layer):
            self.seq.append(glow_block(self.in_dim, 
                                      nn_constructor(self.dim_cf, self.dim_cf, internal_dim_st, nb_layer_st),
                                      nn_constructor(self.dim_cf, self.dim_cf, internal_dim_st, nb_layer_st)))
    
    def reset_parameters(self):
        for m in  self.seq:
            m.reset_parameters()

    def diffeo(self, x):
        return self.seq(x)
    
    def log_jacobian(self, x):
        det = 0
        out = x
        for m in self.seq:
            det += m.log_jacobian(out)
            out = m(out)
        return det
    
    def diffeo_inv(self, x):
        out = x
        for m in reversed(self.seq):
            out = m(out, rev=True)
        return out

    def jacobian(self, x):
        if x.ndim == 1:
            return jacrev(self.forward)(x)
        return vmap(
            jacrev(self.forward)
        )(x)

    def vjp(self, z, v):
        z = z.requires_grad_()
        x = self.diffeo_inv(z) 
        # scalar contraction
        loss_vjp = (x * v).sum()

        # IMPORTANT: create_graph=True
        (vjp,) = torch.autograd.grad(
        loss_vjp,
        z,
        create_graph=True
        )
        return vjp
    
    def autoencoder(self, x):
        n = x.size()[0]
        z = self.diffeo(x)
        z_mean= z.mean(dim=0)
        z_pad = z_mean[self.out_dim:].repeat(n).reshape((n, self.in_dim-self.out_dim)) 
        z_cut = torch.cat((z[:,:self.out_dim],z_pad), dim=1)
        x_pred = self.diffeo_inv(z_cut)
        return z, x_pred
 
    def get_theta(self):
        theta = []
        for m in self.seq:
            theta.append(m.theta)
        return theta
    
    def forward(self, x):
        out = self.diffeo(x)
    
        if out.ndim == 1:
            return out[:self.out_dim]

        return out[:, :self.out_dim]

    def print_parameters(self):
        k=1
        for m in self.seq:
            print("layer "+str(k))
            m.print_parameters()
            k+=1
