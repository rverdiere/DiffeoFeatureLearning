import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_profile(f, g, x, y):
    x_red = g(x)
    fig = plt.figure()
    plt.xlabel('g(x)')
    plt.ylabel('f(x)')
    plt.scatter(x_red.tolist(), y.tolist(), s=2, linewidths=0.5, color="C0", label="Testing set")
    x_0 = torch.tensor(np.linspace(x_red.min().item(), x_red.max().item(),10000)).reshape(10000,1)

    plt.plot(x_0, f(x_0.float()).tolist(), color="C1", label="Profile function")
    plt.legend()
    plt.show()
