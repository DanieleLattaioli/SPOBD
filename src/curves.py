import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from utils import gmse
from unrolling import Unrolling
from pds import PDS

N, T         = 50, 20
MAX_ITER_PDS = 500
SOGLIA       = 1e-4      # sotto questo peso l'arco e' considerato assente
SOGLIA_GAMMA = 1e-3      

T_LIST = [3, 5, 8, 12, 20]
ALPHAS = [0.5, 1, 2, 4, 8]
BETAS  = [0.5, 1, 2, 4, 8]
GAMMAS = [0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
N_TUNE = 30               # campioni usati per il grid search

FIG_DIR = 'figures'
PKL_DIR = 'data_for_plot'

FAMIGLIE = {
    'BA': {
        'etichetta': 'Barabasi-Albert',
        'test':      'generated_data/data_BA50_test.pkl',
        'train':     'generated_data/data_BA50_train.pkl',
        'ckpt_rec':  'trained_models/BA/recurrent_BA50.pt',
        'ckpt_unr':  'trained_models/BA/unrolling_BA50.pt',
        'ckpt_fmt':  'trained_models/BA/unrolling_BA50_T{}.pt',
        'pds':       (4.0, 2.0, 0.1),    # tarato @20 iterazioni
        'pds_conv':  (2.0, 2.0, 0.1),    # tarato a convergenza
    },
    'ER': {
        'etichetta': 'Erdos-Renyi',
        'test':      'generated_data/data_ER50_test.pkl',
        'train':     'generated_data/data_ER50_train.pkl',
        'ckpt_rec':  'trained_models/ER/recurrent_ER50.pt',
        'ckpt_unr':  'trained_models/ER/unrolling_ER50.pt',
        'ckpt_fmt':  'trained_models/ER/unrolling_ER50_T{}.pt',
        'pds':       (4.0, 2.0, 0.1),    # tarato @20 iterazioni
        'pds_conv':  (2.0, 2.0, 0.03),   # tarato a convergenza 
    },
}

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def carica_test(cfg):
    """"
    Carica il dataset di test della famiglia cfg.
    """
    d = pickle.load(open(cfg['test'], 'rb'))
    y = torch.tensor(d['y'], dtype=torch.float32).to(dev)
    w = torch.tensor(d['w'], dtype=torch.float32).to(dev)
    return y, w


def unroll_curve(y_test, w_test, ckpt, shared, T=T, par=(4.0, 2.0, 0.1)):
    """
    Calcola la curva GMSE di un modello Unrolling/Recurrent Unrolling
    """
    a, b, g = par
    model = Unrolling(N=N, T=T, shared=shared, alpha=a, beta=b, gamma=g).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    with torch.no_grad():
        outs = model(y_test)
        curva = [gmse(w_test, o) for o in outs]

    par_out = {
        'alpha': model.alpha.detach().cpu().numpy(),
        'beta':  model.beta.detach().cpu().numpy(),
        'gamma': model.gamma.detach().cpu().numpy(),
    }
    return curva, par_out, outs[-1]


def riduzione(base, nuovo):
    """"
    Calcola la riduzione percentuale di GMSE da base a nuovo
    """
    return 100 * (base - nuovo) / base


def gradi(W_batch, W_ref=None, soglia=SOGLIA):
    """
    Estrae i gradi da un batch di stime.
    Se W_ref e' fornita, ogni grafo viene sparsificato tenendo i K archi
    piu' pesanti, con K pari al numero di archi del grafo di riferimento.
    """
    W_batch = W_batch.detach().cpu().numpy()
    if W_ref is not None:
        W_ref = W_ref.detach().cpu().numpy()

    tutti, sigmas, massimi = [], [], []
    for i, w in enumerate(W_batch):
        if W_ref is None:
            binario = (w > soglia).astype(int)
        else:
            K = int((W_ref[i] > soglia).sum())
            idx = np.argsort(w)[::-1][:K]
            binario = np.zeros_like(w, dtype=int)
            binario[idx] = 1
        A = squareform(binario)
        k = A.sum(1)
        tutti.append(k)
        sigmas.append(k.std())
        massimi.append(k.max())
    return np.concatenate(tutti), np.array(sigmas), np.array(massimi)


def compute_confronto(fam):
    """
    Calcola le curve GMSE di PDS, Recurrent Unrolling e Unrolling
    """
    cfg = FAMIGLIE[fam]
    print(f'\n=== confronto @T={T} — famiglia {fam} ({cfg["etichetta"]}) ===')
    y_test, w_test = carica_test(cfg)

    # Curve GMSE
    a, b, g = cfg['pds']
    ac, bc, gc = cfg['pds_conv']
    with torch.no_grad():
        curva_pds   = PDS.pds_curve(y_test, a, b, g, MAX_ITER_PDS, w_test)
        curva_pds_c = PDS.pds_curve(y_test, ac, bc, gc, MAX_ITER_PDS, w_test)

    print(f'PDS@20   : min {min(curva_pds):.4f} a iter {curva_pds.index(min(curva_pds))+1}, 'f'finale {curva_pds[-1]:.4f}')
    print(f'PDS@conv : min {min(curva_pds_c):.4f} a iter {curva_pds_c.index(min(curva_pds_c))+1}, 'f'finale {curva_pds_c[-1]:.4f}')

    curva_rec, par_rec, w_rec = unroll_curve(y_test, w_test, cfg['ckpt_rec'], True,  par=cfg['pds'])
    curva_unr, par_unr, w_unr = unroll_curve(y_test, w_test, cfg['ckpt_unr'], False, par=cfg['pds'])

    print(f'Recurrent @{T} layer: {curva_rec[-1]:.4f}')
    print(f'Unrolling @{T} layer: {curva_unr[-1]:.4f}')

    print()
    print(f'riduzione GMSE Unrolling vs PDS@20   : {riduzione(curva_pds[T-1],   curva_unr[-1]):.1f}%')
    print(f'riduzione GMSE Unrolling vs PDS@conv : {riduzione(min(curva_pds_c), curva_unr[-1]):.1f}%')
    print(f'riduzione GMSE Unrolling vs Recurrent: {riduzione(curva_rec[-1],    curva_unr[-1]):.1f}%')

    # Distribuzione dei gradi
    with torch.no_grad():
        w_pds = PDS.pds_curve(y_test, a, b, g, T, w_test, return_w=True)

    serie = {'groundtruth': w_test, 'PDS': w_pds, 'Recurrent': w_rec, 'Unrolling': w_unr}

    print()
    stat = {}
    for nome, W in serie.items():
        ref = None if nome == 'groundtruth' else w_test
        k, s, mx = gradi(W, W_ref=ref)
        stat[nome] = {'gradi': k, 'sigma': s, 'max': mx}
        ic_s  = 1.96 * s.std()  / np.sqrt(len(s))
        ic_mx = 1.96 * mx.std() / np.sqrt(len(mx))
        print(f'{nome:12s}  <k>={k.mean():5.2f}   '
              f'sigma_k={s.mean():5.2f}±{ic_s:.2f}   '
              f'k_max={mx.mean():5.2f}±{ic_mx:.2f}')

    risultati = {
        'famiglia': fam,
        'etichetta': cfg['etichetta'],
        'pds': curva_pds,
        'pds_conv': curva_pds_c,
        'recurrent': curva_rec,
        'unrolling': curva_unr,
        'par_recurrent': par_rec,
        'par_unrolling': par_unr,
        'pds_params': cfg['pds'],
        'pds_conv_params': cfg['pds_conv'],
        'stat': stat,
    }
    path = os.path.join(PKL_DIR, f'curves_{fam}50.pkl')
    pickle.dump(risultati, open(path, 'wb'))
    print(f'\nsalvato: {path}')
    return risultati

def compute_depth(fam):
    """ 
    Calcola la curva GMSE vs T per PDS, Recurrent Unrolling e Unrolling
    """
    cfg = FAMIGLIE[fam]
    print(f'\n=== GMSE vs T — famiglia {fam} ({cfg["etichetta"]}) ===')
    y_test, w_test = carica_test(cfg)

    dtr = pickle.load(open(cfg['train'], 'rb'))
    Y_tr, W_tr = dtr['y'], dtr['w']

    res = {'famiglia': fam, 'etichetta': cfg['etichetta'],
           'T': T_LIST, 'unrolling': [], 'pds': [], 'par': {}}

    for Tn in T_LIST:
        model = Unrolling(N=N, T=Tn, shared=False).to(dev)
        model.load_state_dict(torch.load(cfg['ckpt_fmt'].format(Tn), map_location=dev))
        model.eval()
        with torch.no_grad():
            g_unr = gmse(w_test, model(y_test)[-1])
        res['unrolling'].append(g_unr)
        res['par'][Tn] = {
            'alpha': model.alpha.detach().cpu().numpy(),
            'beta':  model.beta.detach().cpu().numpy(),
            'gamma': model.gamma.detach().cpu().numpy(),
        }

        solver = PDS(alpha=2, beta=2, gamma=0.1)
        solver.tune(Y_tr, W_tr, ALPHAS, BETAS, GAMMAS, n_samples=N_TUNE, max_iter=Tn)
        with torch.no_grad():
            w_pds = PDS.pds_curve(y_test, solver.alpha, solver.beta, solver.gamma, Tn, w_test, return_w=True)
            g_pds = gmse(w_test, w_pds)
        res['pds'].append(g_pds)

        print(f'T={Tn:2d}   PDS {g_pds:.4f}   Unrolling {g_unr:.4f}   '
              f'riduzione {riduzione(g_pds, g_unr):5.1f}%   '
              f'(PDS: a={solver.alpha}, b={solver.beta}, g={solver.gamma})')

    print()
    for Tn in T_LIST:
        gg = res['par'][Tn]['gamma']
        attivi = np.where(gg > SOGLIA_GAMMA)[0]
        eff = int(attivi.max()) + 1 if len(attivi) else 0
        print(f'T={Tn:2d}   layer attivi (gamma > {SOGLIA_GAMMA:g}): {eff}')

    path = os.path.join(PKL_DIR, f'depth_{fam}50.pkl')
    pickle.dump(res, open(path, 'wb'))
    print(f'\nsalvato: {path}')
    return res


# Grafici
def plot_curve_gmse(risultati):
    """
    Curve GMSE di PDS, Recurrent Unrolling e Unrolling
    """
    curva_pds, curva_pds_c = risultati['pds'], risultati['pds_conv']
    curva_rec, curva_unr   = risultati['recurrent'], risultati['unrolling']

    it_pds = range(1, len(curva_pds) + 1)
    it_unr = range(1, T + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(it_pds, curva_pds,   '--', color='0.6', label='PDS (tunato @20 iter)')
    ax.plot(it_pds, curva_pds_c, '-',  color='0.3', label='PDS (tunato a convergenza)')
    ax.plot(it_unr, curva_rec, 'o-', ms=3, label='Recurrent Unrolling')
    ax.plot(it_unr, curva_unr, 's-', ms=3, label='Unrolling')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.05, 2)
    ax.set_xlabel('iterazioni / layer')
    ax.set_ylabel('GMSE (test)')
    ax.set_title(risultati['etichetta'], fontsize=10)
    ax.grid(alpha=.3, which='both')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_istogramma_gradi(risultati):
    """
    Istogramma dei gradi di PDS, Recurrent Unrolling e Unrolling
    """
    stat = risultati['stat']
    kmax = max(s['gradi'].max() for s in stat.values())
    bins = np.arange(0, kmax + 2) - 0.5
    centri = np.arange(0, kmax + 1)

    stili = {'groundtruth': dict(color='k', lw=2),
             'PDS':         dict(color='0.5', ls='--'),
             'Recurrent':   dict(color='tab:blue', ls='-.'),
             'Unrolling':   dict(color='tab:orange', ls='-')}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nome, s in stat.items():
        h, _ = np.histogram(s['gradi'], bins=bins, density=True)
        ax.plot(centri, h, label=nome, **stili[nome])

    ax.set_xlabel('grado $k$')
    ax.set_ylabel('frazione di nodi')
    ax.set_xlim(0, kmax)
    ax.set_title(risultati['etichetta'], fontsize=10)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_istogramma_gradi_confronto(cache_confronto):
    """
    Istogrammi dei gradi affiancati, uno per famiglia (stesso asse y)
    """
    stili = {'groundtruth': dict(color='k', lw=2),
             'PDS':         dict(color='0.5', ls='--'),
             'Recurrent':   dict(color='tab:blue', ls='-.'),
             'Unrolling':   dict(color='tab:orange', ls='-')}

    ordine = ['ER', 'BA']
    fams = sorted(cache_confronto, key=lambda f: ordine.index(f) if f in ordine else len(ordine))
    fig, axes = plt.subplots(1, len(fams), figsize=(7 * len(fams), 4.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, fam in zip(axes, fams):
        risultati = cache_confronto[fam]
        stat = risultati['stat']
        kmax = max(s['gradi'].max() for s in stat.values())
        bins = np.arange(0, kmax + 2) - 0.5
        centri = np.arange(0, kmax + 1)

        for nome, s in stat.items():
            h, _ = np.histogram(s['gradi'], bins=bins, density=True)
            ax.plot(centri, h, label=nome, **stili[nome])

        ax.set_xlabel('grado $k$')
        ax.set_xlim(0, kmax)
        ax.set_title(risultati['etichetta'], fontsize=10)
        ax.grid(alpha=.3)
        ax.legend()

    axes[0].set_ylabel('frazione di nodi')
    fig.tight_layout()
    return fig


def plot_parametri_layer(risultati):
    """"
    Parametri alpha, beta, gamma layer per layer di Unrolling e Recurrent Unrolling
    """
    par_unr, par_rec = risultati['par_unrolling'], risultati['par_recurrent']
    layer = np.arange(1, T + 1)

    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for ax, chiave, simbolo in zip(axes,
                                   ['alpha', 'beta', 'gamma'],
                                   [r'$\alpha^{(t)}$', r'$\beta^{(t)}$', r'$\gamma^{(t)}$']):
        ax.plot(layer, par_unr[chiave], 's-', ms=4,color='tab:orange', label='Unrolling')
        ax.axhline(par_rec[chiave][0], color='tab:blue', ls='-.', lw=1, label='Recurrent (condiviso)')
        ax.set_ylabel(simbolo)
        ax.grid(alpha=.3)

    axes[2].set_yscale('log')
    axes[2].set_xlabel('indice del layer $t$')
    axes[0].legend(fontsize=8)
    axes[0].set_title(risultati['etichetta'], fontsize=10)
    axes[0].set_xticks(layer[::2])
    fig.tight_layout()
    return fig


def plot_gmse_vs_t(res):
    """"
    Curve GMSE vs T per PDS, Recurrent Unrolling e Unrolling
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res['T'], res['pds'], 'o--', color='0.4', label='PDS (tarato per ogni $T$)')
    ax.plot(res['T'], res['unrolling'], 's-', color='tab:orange', label='Unrolling')

    for Tn, gu in zip(res['T'], res['unrolling']):
        ax.annotate(f'{gu:.3f}', (Tn, gu), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=7, color='tab:orange',
                    annotation_clip=False,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

    ax.set_yscale('log')
    ax.set_xticks(res['T'])
    ax.set_xlabel('numero di layer / iterazioni $T$')
    ax.set_ylabel('GMSE (test)')
    ax.set_title(res['etichetta'], fontsize=10)
    ax.grid(alpha=.3, which='both')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin * 0.7, ymax)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_confronto_famiglie(cache_confronto):
    """
    Curve GMSE vs iterazioni per tutte le famiglie sullo stesso asse
    """
    colori = {'BA': 'tab:orange', 'ER': 'tab:green', 'WS': 'tab:purple'}
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for fam, r in cache_confronto.items():
        c = colori.get(fam, None)
        ax.plot(range(1, len(r['pds_conv']) + 1), r['pds_conv'],
                '--', color=c, alpha=.6, label=f'PDS — {fam}')
        ax.plot(range(1, T + 1), r['unrolling'],
                's-', ms=3, color=c, label=f'Unrolling — {fam}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.05, 2)
    ax.set_xlabel('iterazioni / layer')
    ax.set_ylabel('GMSE (test)')
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def tabella_famiglie(cache_confronto):
    """
    Stampa la tabella riassuntiva GMSE e riduzioni per famiglia
    """
    fams = list(cache_confronto)
    print('\n=== tabella riassuntiva ===')
    print(f"{'':14s}" + ''.join(f'{f:>12s}' for f in fams))
    for etichetta, chiave in [('PDS @20', None), ('Recurrent', 'recurrent'),
                              ('Unrolling', 'unrolling')]:
        vals = []
        for f in fams:
            r = cache_confronto[f]
            v = r['pds'][T-1] if chiave is None else r[chiave][-1]
            vals.append(v)
        print(f'{etichetta:14s}' + ''.join(f'{v:12.4f}' for v in vals))

    print(f"\n{'riduzione %':14s}" + ''.join(f'{f:>12s}' for f in fams))
    rid_pds = [riduzione(cache_confronto[f]['pds'][T-1],
                         cache_confronto[f]['unrolling'][-1]) for f in fams]
    print(f'{"Unr. vs PDS":14s}' + ''.join(f'{v:11.1f}%' for v in rid_pds))
    rid_rec = [riduzione(cache_confronto[f]['recurrent'][-1],
                         cache_confronto[f]['unrolling'][-1]) for f in fams]
    print(f'{"Unr. vs Rec":14s}' + ''.join(f'{v:11.1f}%' for v in rid_rec))


FIG_SPECS = {
    'curve_gmse':       ('confronto', plot_curve_gmse,       'curve_gmse_{fam}.png'),
    'istogramma_gradi': ('confronto', plot_istogramma_gradi, 'istogramma_gradi_{fam}.png'),
    'parametri_layer':  ('confronto', plot_parametri_layer,  'parametri_layer_{fam}.png'),
    'gmse_vs_t':        ('depth',     plot_gmse_vs_t,        'gmse_vs_T_{fam}.png'),
}

COMPUTE_FN = {'confronto': compute_confronto, 'depth': compute_depth}

def main(grafici, famiglie, confronto_famiglie=True):
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PKL_DIR, exist_ok=True)
    cache = {}          

    for fam in famiglie:
        for nome in grafici:
            gruppo, plot_fn, filename = FIG_SPECS[nome]
            chiave = (gruppo, fam)
            if chiave not in cache:
                cache[chiave] = COMPUTE_FN[gruppo](fam)
            fig = plot_fn(cache[chiave])
            path = os.path.join(FIG_DIR, filename.format(fam=fam))
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f'salvato: {path}')

    conf = {f: cache[('confronto', f)] for f in famiglie if ('confronto', f) in cache}
    if confronto_famiglie and len(conf) > 1:
        fig = plot_confronto_famiglie(conf)
        path = os.path.join(FIG_DIR, 'confronto_famiglie.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'salvato: {path}')

        fig = plot_istogramma_gradi_confronto(conf)
        path = os.path.join(FIG_DIR, 'istogramma_gradi_confronto.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'salvato: {path}')

        tabella_famiglie(conf)


if __name__ == '__main__':
    GRAFICI = [
        'curve_gmse',
        'istogramma_gradi',
        'parametri_layer',
        'gmse_vs_t'
    ]
    FAM = ['BA', 'ER']

    main(GRAFICI, FAM)