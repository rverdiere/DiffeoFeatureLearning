import torch
import torch.linalg as alg
from torch import vmap
from torch.func import jacrev, grad, jvp, vjp
from math import sqrt

def compute_errors(x,y):
    mse = MSE(x, y).item()
    nrmse = NRMSE(x, y).item()
    rl1 = RL1(x, y).item()
    rl2 = RL2(x, y).item()
    return [mse, nrmse, rl1, rl2]

def H1Loss(x, y, grad_x, grad_y):
    return MSE(x,y)+MSE(grad_x, grad_y)

def MSE(x, y):
    return torch.mean(alg.norm(x-y, ord=2, axis=1)**2)

def NRMSE(x, y):
    val= torch.sqrt(MSE(x,y))
    val/= (torch.max(x)-torch.min(x))
    return val

def RL2(x,y):
    val= MSE(x,y)
    val /= torch.mean(alg.norm(x, ord=2, axis=1)**2)
    return torch.sqrt(val)

def RL1(x, y):
    val= torch.mean(alg.norm(x-y, ord=1, axis=1))
    val /= torch.mean(alg.norm(x, ord=1, axis=1))
    return val 

def norm_l2(y, y_pred):
    u_norm = torch.mean(alg.norm(y, dim=1)**2)
    return MSE(y,y_pred)/u_norm

def poincare_IS(g, x, x_grad):
    jacobian = g.jacobian(x)
    grad_norm = torch.mean(alg.norm(x_grad, dim=(1,2))**2)
    
    #projector computation
    jacobian_t = torch.transpose(jacobian, 1,2)
    inv = torch.inverse(jacobian@jacobian_t)
    proj = jacobian_t@inv@jacobian

    proj_x_grad = torch.bmm(proj,x_grad)
    pi_loss= torch.mean(alg.norm(x_grad-proj_x_grad, axis=(-1,-2))**2)/grad_norm
    
    return pi_loss


def poincare_IS_chol(g, x, x_grad, eps=1e-6):
    """
    x_grad: (B, d, q)
    jacobian: (B, m, d)
    """
    
    jacobian = g.jacobian(x)
    B, m, d = jacobian.shape
    grad_norm = torch.mean(torch.sum(x_grad**2, dim=(1,2)))

    # G = JJ^T
    G = torch.bmm(jacobian, jacobian.transpose(1, 2))
    # regularization for stability
    G = G + eps * torch.eye(m, device=jacobian.device).unsqueeze(0)
    # Cholesky factorization: G = L L^T
    L = torch.linalg.cholesky(G)
    # b = J x_grad
    b = torch.bmm(jacobian, x_grad)  # (B,m,1)
    # solve L y = b
    y = torch.linalg.solve_triangular(L, b, upper=False)
    # solve L^T c = y
    c = torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)
    # projection energy term: b^T G^{-1} b
    proj_energy = torch.sum(b.squeeze(-1) * c.squeeze(-1), dim=1)

    loss = 1 - torch.mean(proj_energy).grad_norm

    return loss


def poincare_FS(g, x, x_grad, cov_inv, lamb=0.01):
    #Compute grad norm to normalize the Poincaré loss
    z = g.diffeo(x)
    m = g.out_dim
    logdet = g.log_jacobian(x)
    
    grad_norm = (x_grad**2).sum(dim=1).mean()
    
    # Compute Kullback-Leibler divergence term
    norm_phi_sq = torch.diag(z@cov_inv@z.t())
    DKL =((norm_phi_sq)/2-logdet).mean()
    
    # Compute Feature Space Poincaré loss from VJP
    _,_,q = x_grad.size()
    PIloss = 0
    for k in range(q):
        v = x_grad[:,:,k]
        vjp_perp = g.vjp(z,v)[:,m:]
        PIloss+=(vjp_perp ** 2).sum(dim=1).mean()
    
    return PIloss/grad_norm+lamb*DKL

def poincare_FS_ae(g, x, cov_inv, lamb=1e-2):
    z = g.diffeo(x)
    
    compute_batch_jacobian = vmap(jacrev(g.diffeo_inv), in_dims=(0))
    jacobian = compute_batch_jacobian(z)[:, g.out_dim:, :]
    PI= (alg.norm(jacobian, dim=(1,2))**2).mean()
    
    norm_phi_sq = torch.diag(z@cov_inv@z.t())
    logdet = g.log_jacobian(x)
    DKL =((norm_phi_sq)/2-logdet).mean()
    
    return PI+lamb*DKL
