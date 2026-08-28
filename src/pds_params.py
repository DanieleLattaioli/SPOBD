"""Parametri PDS (alpha, beta, gamma) tunati via PDS.tune() su ciascun tipo di
grafo, cosi' da non dover rifare il grid search ogni volta che si cambia
GRAPH_TYPE. Le due varianti:

- '20iter'     : tune con max_iter=20 (stesso budget di iterazioni di un
                 Unrolling a T=20 layer)
- 'convergenza': tune con max_iter=500 (PDS lasciato convergere)

Tuning eseguito con PDS.tune(Y, W, ALPHAS, BETAS, GAMMAS, n_samples=30, ...)
sul train set, alphas/betas/gammas come in curves.py.
"""

PDS_PARAMS = {
    'BA': {
        '20iter':      {'alpha': 4.0, 'beta': 2.0, 'gamma': 0.1},
        'convergenza': {'alpha': 2.0, 'beta': 2.0, 'gamma': 0.1},
    },
    'ER': {
        '20iter':      {'alpha': 4.0, 'beta': 2.0, 'gamma': 0.1},
        'convergenza': {'alpha': 2.0, 'beta': 2.0, 'gamma': 0.03},
    },
}
