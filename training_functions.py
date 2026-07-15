import torch
from torch.func import vmap
import copy
from losses import poincare_IS, poincare_IS_chol, poincare_FS

def multistart(nb_start, g, dataloader, loss_function, max_epoch=300, learning_rate=0.01, print_freq=100, sig=0.01, lamb_KL=0.01):
    g_list_ms = []
    f_list_ms = []
    losses = []
    for k in range(nb_start):
        print(f"Run {k}:")
        g_cur =copy.deepcopy(g)
        g_cur.reset_parameters()
        
        (g_cur, loss) = train_featuremap(g_cur, dataloader, loss_function, max_epoch, learning_rate, print_freq, lamb_KL=lamb_KL)
        g_list_ms.append(g_cur)
        losses.append(loss.item())

    m = min(losses)
    g_res = g_list_ms[losses.index(m)]
    return g_res


def train_profilefunc(f, g, dataloader, loss_function, max_epoch=2000, learning_rate=0.01, print_freq=100):
    
    optimizer = torch.optim.Adam(f.parameters(), lr=learning_rate)

    for epoch in range(1, max_epoch + 1):
        for x_batch, grad_x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            with torch.no_grad():
                x_batch_red = g(x_batch)
            y_pred = f(x_batch_red) 
            loss = loss_function(y_pred, y_batch)
            loss.backward(retain_graph=True)
            optimizer.step()
        if print_freq is not None and epoch % print_freq == 0:
            print(f"loss: {loss:>6f}  [{epoch:>5d}/{max_epoch:>5d}]")
            
    return f

def train_featuremap(g,dataloader,loss_function,max_epoch=3000,learning_rate=0.01,print_freq=500,sig=0.1, lamb_KL=0.01):
    g.train()

    optimizer = torch.optim.Adam([
        {"params": g.parameters(), "lr": learning_rate}
    ])

    #KL divergence parameters
    cov_vec = torch.cat((torch.ones(g.out_dim), (sig**2)*torch.ones(g.in_dim-g.out_dim)))
    cov_inv_mat = torch.diag(1/cov_vec)
    
    for epoch in range(1, max_epoch + 1):
        for x_batch, grad_x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            if loss_function == poincare_IS or loss_function == poincare_IS_chol:
                loss = loss_function(g, x_batch, grad_x_batch)
            elif loss_function == poincare_FS:
                loss = loss_function(g, x_batch, grad_x_batch, cov_inv_mat, lamb_KL)
            else:
                raise ValueError("Unsupported loss function")
            
            loss.backward()
            optimizer.step()

        if print_freq is not None and epoch % print_freq == 0:
            # avoid unnecessary syncs
            print(f"loss: {loss.item():>6f}  [{epoch:>5d}/{max_epoch:>5d}]")

    return g, loss
