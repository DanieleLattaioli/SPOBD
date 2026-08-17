import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import euclidean_distances

OUT_PKL_DIR = 'generated_data'

def sample(N=50, n_signals=3000, rng=None):
    # Generazione grafo Barabasi-Albert
    G = nx.barabasi_albert_graph(N, 2, seed=int(rng.integers(1e9)))

    # Definizioni matrici A, W ed L
    A = nx.to_numpy_array(G)
    S = rng.normal(0, 0.1, (N,N))
    W = np.exp(S) * A
    W = (W + W.T) / 2
    W = W * N / W.sum()
    L = np.diag(W.sum(1)) - W

    # Generazione segnali
    P = L + 1e-4 * np.eye(N) # Precision matrix
    cov = np.linalg.inv(P)
    X = rng.multivariate_normal(np.zeros(N), cov, n_signals) # Segnali su grafo (n, N)

    # Half-vectorisation
    edM = euclidean_distances(X.T, squared=True) / n_signals # (N, N)
    y = squareform(edM, checks=False) # (N(N-1)/2, 1)
    w = squareform(W, checks=False)

    return y, w

rng = np.random.default_rng(42)

for nome, n_campioni in [('train', 8000), ('val', 2000), ('test', 64)]:
    dati = []
    for i in range(n_campioni):
        if i % 1000 == 0:
            print(f'{nome}: {i}/{n_campioni}', flush=True)
        dati.append(sample(rng=rng))
    Y, Wv = zip(*dati)
    pickle.dump({'y': np.array(Y), 'w': np.array(Wv)},
                open(f'{OUT_PKL_DIR}/data_BA50_{nome}.pkl', 'wb'))
    print(f'{nome} completato', flush=True)
