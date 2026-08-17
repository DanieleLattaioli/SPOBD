"""Grafici dei risultati sperimentali (PDS, Unrolling, Recurrent Unrolling).

Ogni grafico e' una funzione indipendente che restituisce una Figure: per
aggiungerne uno nuovo basta scrivere una funzione sullo stesso modello e
richiamarla in main().
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

# ---------------- config ----------------
DATA = 'data1_BA20.pkl'
N_TR, N_VA = 4000, 500
PDS_ALPHA, PDS_BETA, PDS_GAMMA = 2, 2, 0.1
N_ITER_PDS = 500          # per il valore di PDS a convergenza
N_LAYERS = 20             # layer degli unrolling / iterazioni PDS mostrate nel confronto
FIG_DIR = 'figures'

RESULTS_FILES = {
    'unrolling':     'results_unrolling_BA20.pkl',
    'recurrent_log': 'results_recurrent_BA20.pkl',
    'recurrent_raw': 'results_recurrent_BA20_raw.pkl',
}
COLORS = {
    'pds':           '#2a78d6',
    'unrolling':     '#eb6834',
    'recurrent_log': '#1baf7a',
    'recurrent_raw': '#eda100',
}
LABELS = {
    'pds':           'PDS',
    'unrolling':     'Unrolling',
    'recurrent_log': 'Recurrent (log-param)',
    'recurrent_raw': 'Recurrent (raw)',
}


# ---------------- caricamento dati ----------------
def load_results():
    return {key: pickle.load(open(path, 'rb')) for key, path in RESULTS_FILES.items()}


def make_D_np(N):
    E = N * (N - 1) // 2
    D = np.zeros((N, E))
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            D[i, k] = D[j, k] = 1.
            k += 1
    return D


def gmse_np(wh, w):
    return ((wh - w) ** 2).sum() / (w ** 2).sum()


def pds_gmse_trace(n_iter=N_ITER_PDS, alpha=PDS_ALPHA, beta=PDS_BETA, gamma=PDS_GAMMA):
    """GMSE medio di PDS ad ogni iterazione, sullo stesso test set di train.py."""
    d = pickle.load(open(DATA, 'rb'))
    Y, W = d['y'], d['w']
    E = Y.shape[1]
    N = round((1 + np.sqrt(1 + 8 * E)) / 2)
    D = make_D_np(N)
    Yte, Wte = Y[N_TR + N_VA:], W[N_TR + N_VA:]

    trace = np.zeros(n_iter)
    for i in range(len(Yte)):
        y, w_true = Yte[i], Wte[i]
        w = np.zeros(E)
        v = np.zeros(N)
        for t in range(n_iter):
            r1 = w - gamma * (2 * beta * w + 2 * y + D.T @ v)
            r2 = v + gamma * (D @ w)
            p1 = np.maximum(r1, 0)
            p2 = (r2 - np.sqrt(r2 ** 2 + 4 * alpha * gamma)) / 2
            q1 = p1 - gamma * (2 * beta * p1 + 2 * y + D.T @ p2)
            q2 = p2 + gamma * (D @ p1)
            w = w - r1 + q1
            v = v - r2 + q2
            trace[t] += gmse_np(w, w_true)
    trace /= len(Yte)
    return trace


# ---------------- singoli grafici ----------------
def plot_training_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ['unrolling', 'recurrent_log', 'recurrent_raw']:
        hist = results[key]['hist']
        epochs = np.arange(len(hist))
        style = dict(color=COLORS[key], label=LABELS[key], linewidth=1.8)
        if key == 'recurrent_raw':
            style['linestyle'] = '--'
        axes[0].plot(epochs, [h[0] for h in hist], **style)
        axes[1].plot(epochs, [h[1] for h in hist], **style)

    axes[0].set(xlabel='epoca', ylabel='train loss', title='Train loss')
    axes[1].set(xlabel='epoca', ylabel='val GMSE', yscale='log', title='Validation GMSE')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_gmse_per_layer(results, pds_trace):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    layers = np.arange(1, N_LAYERS + 1)

    ax.plot(layers, pds_trace[:N_LAYERS], color=COLORS['pds'], label=LABELS['pds'], linewidth=1.8)
    for key in ['unrolling', 'recurrent_log', 'recurrent_raw']:
        style = dict(color=COLORS[key], label=LABELS[key], linewidth=1.8)
        if key == 'recurrent_raw':
            style['linestyle'] = '--'
        ax.plot(layers, results[key]['gmse_layer'], **style)

    ax.set(xlabel='layer / iterazione', ylabel='GMSE', yscale='log',
           title='GMSE per layer/iterazione (test set)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    return fig


def plot_hyperparameters(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    layers = np.arange(1, N_LAYERS + 1)

    for ax, name in zip(axes, ['alpha', 'beta', 'gamma']):
        ax.plot(layers, results['unrolling'][name], color=COLORS['unrolling'],
                label=LABELS['unrolling'], linewidth=1.8)
        ref = results['recurrent_log'][name][0]
        ax.axhline(ref, color=COLORS['recurrent_log'], linestyle='--',
                   label=LABELS['recurrent_log'], linewidth=1.5)
        ax.set(xlabel='layer', ylabel=name, title=name)
        if name == 'gamma':
            ax.set_yscale('log')
        ax.grid(alpha=0.25)

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_final_comparison(results, pds_final):
    order = ['pds', 'recurrent_log', 'recurrent_raw', 'unrolling']
    values = [pds_final] + [results[k]['gmse_layer'][-1] for k in order[1:]]
    colors = [COLORS[k] for k in order]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar([LABELS[k] for k in order], values, color=colors)
    ax.bar_label(bars, fmt='%.4f', padding=3)
    ax.set(ylabel='GMSE test', title='Confronto finale')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------- main ----------------
def main():
    results = load_results()
    pds_trace = pds_gmse_trace()

    figures = {
        'training_curves':  plot_training_curves(results),
        'gmse_per_layer':   plot_gmse_per_layer(results, pds_trace),
        'hyperparameters':  plot_hyperparameters(results),
        'final_comparison': plot_final_comparison(results, pds_trace[-1]),
    }

    os.makedirs(FIG_DIR, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=150)
        print(f'salvato {FIG_DIR}/{name}.png')

    plt.show()


if __name__ == '__main__':
    main()
