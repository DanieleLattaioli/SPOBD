import torch

def make_D(N):
    E = N*(N-1)//2
    D = torch.zeros((N, E)); 
    k = 0
    for i in range(N):
        for j in range(i+1, N):
            D[i,k] = D[j,k] = 1.; 
            k += 1
    return D

def prox2(r2, alfa, gamma):
    return (r2 - torch.sqrt(r2**2 + 4 * alfa * gamma)) / 2

def gmse(W, W_pred):
    return (((W - W_pred) ** 2).sum(1) / (W ** 2).sum(1)).mean().item()