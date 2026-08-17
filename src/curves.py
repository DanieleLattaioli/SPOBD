import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

from utils import gmse, make_D
from unrolling import Unrolling
from pds import PDS

N, T        = 50, 20
TEST        = 'generated_data/data_BA50_test.pkl'
CKPT_REC    = 'trained_models/recurrent_BA50.pt'
CKPT_UNR    = 'trained_models/unrolling_BA50.pt'

PDS_ALPHA, PDS_BETA, PDS_GAMMA = 4.0, 2.0, 0.1
PDS_C_ALPHA, PDS_C_BETA, PDS_C_GAMMA = 2.0, 2.0, 0.1

MAX_ITER_PDS = 500
SOGLIA       = 1e-4      # sotto questo peso l'arco è considerato assente
OUT_PKL      = 'data_for_plot/curves_BA50.pkl'
OUT_PNG      = 'figures/curve_gmse.png'
OUT_PNG_DEG  = 'figures/istogramma_gradi.png'
OUT_PNG_PAR = 'figures/parametri_layer.png'

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# Caricamento test set
d = pickle.load(open(TEST, 'rb'))
y_test = torch.tensor(d['y'], dtype=torch.float32).to(dev)
w_test = torch.tensor(d['w'], dtype=torch.float32).to(dev)

# Calcolo curve GMSE per PDS (due tarature)
with torch.no_grad():
    curva_pds   = PDS.pds_curve(y_test, PDS_ALPHA, PDS_BETA, PDS_GAMMA,
                                MAX_ITER_PDS, w_test)
    curva_pds_c = PDS.pds_curve(y_test, PDS_C_ALPHA, PDS_C_BETA, PDS_C_GAMMA,
                                MAX_ITER_PDS, w_test)

print(f'PDS@20   : min {min(curva_pds):.4f} a iter {curva_pds.index(min(curva_pds))+1}, '
      f'finale {curva_pds[-1]:.4f}')
print(f'PDS@conv : min {min(curva_pds_c):.4f} a iter {curva_pds_c.index(min(curva_pds_c))+1}, '
      f'finale {curva_pds_c[-1]:.4f}')


# Funzione per calcolo curva GMSE per Recurrent Unrolling e Unrolling
def unroll_curve(ckpt, shared):
    model = Unrolling(N=N, T=T, shared=shared,
                      alpha=PDS_ALPHA, beta=PDS_BETA, gamma=PDS_GAMMA).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    with torch.no_grad():
        outs = model(y_test)
        curva = [gmse(w_test, o) for o in outs]
    par = {
        'alpha': model.alpha.detach().cpu().numpy(),
        'beta':  model.beta.detach().cpu().numpy(),
        'gamma': model.gamma.detach().cpu().numpy(),
    }
    return curva, par, outs[-1]


# Calcolo curve GMSE per Recurrent Unrolling e Unrolling
curva_rec, par_rec, w_rec = unroll_curve(CKPT_REC, shared=True)
curva_unr, par_unr, w_unr = unroll_curve(CKPT_UNR, shared=False)

print(f'Recurrent @{T} layer: {curva_rec[-1]:.4f}')
print(f'Unrolling @{T} layer: {curva_unr[-1]:.4f}')


# ---------------- iterazioni PDS equivalenti ----------------
def iter_equivalenti(curva, soglia):
    """Prima iterazione in cui PDS scende sotto la soglia (None se mai)."""
    for i, g in enumerate(curva, 1):
        if g <= soglia:
            return i
    return None


# confronto con PDS nella sua configurazione migliore (tunato a convergenza)
eq_rec = iter_equivalenti(curva_pds_c, curva_rec[-1])
eq_unr = iter_equivalenti(curva_pds_c, curva_unr[-1])
print(f'\nPDS raggiunge il Recurrent dopo {eq_rec} iterazioni')
print(f'PDS raggiunge l\'Unrolling dopo {eq_unr} iterazioni')

 
def riduzione(base, nuovo):
    return 100 * (base - nuovo) / base
 
print()
print(f'riduzione GMSE Unrolling vs PDS@20   : '
      f'{riduzione(curva_pds[T-1],   curva_unr[-1]):.1f}%')
print(f'riduzione GMSE Unrolling vs PDS@conv : '
      f'{riduzione(min(curva_pds_c), curva_unr[-1]):.1f}%')
print(f'riduzione GMSE Unrolling vs Recurrent: '
      f'{riduzione(curva_rec[-1],    curva_unr[-1]):.1f}%')


# ---------------- distribuzione dei gradi ----------------
# stima di PDS al termine di T iterazioni, nella taratura @20
with torch.no_grad():
    w_pds = PDS.pds_curve(y_test, PDS_ALPHA, PDS_BETA, PDS_GAMMA,
                          T, w_test, return_w=True)


def gradi(W_batch, W_ref=None, soglia=SOGLIA):
    """
    Estrae i gradi da un batch di stime.
    Se W_ref è fornita, ogni grafo viene sparsificato tenendo i K archi
    più pesanti, con K pari al numero di archi del grafo di riferimento.
    Altrimenti si usa la soglia assoluta.
    Restituisce (gradi concatenati, sigma_k per grafo, grado max per grafo).
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


serie = {
    'groundtruth': w_test,
    'PDS':         w_pds,
    'Recurrent':   w_rec,
    'Unrolling':   w_unr,
}

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


# errore relativo sui gradi
D = make_D(N).to(dev)


def errore_gradi(W_pred, W_true):
    dp = W_pred @ D.T
    dt = W_true @ D.T
    return (((dp - dt) ** 2).sum(1) / (dt ** 2).sum(1)).mean().item()


print()
err_gradi = {}
for nome in ['PDS', 'Recurrent', 'Unrolling']:
    err_gradi[nome] = errore_gradi(serie[nome], w_test)
    print(f'errore relativo sui gradi, {nome:10s}: {err_gradi[nome]:.4f}')


# ---------------- salvataggio ----------------
pickle.dump({'pds': curva_pds,
             'pds_conv': curva_pds_c,
             'recurrent': curva_rec,
             'unrolling': curva_unr,
             'par_recurrent': par_rec,
             'par_unrolling': par_unr,
             'pds_params': (PDS_ALPHA, PDS_BETA, PDS_GAMMA),
             'pds_conv_params': (PDS_C_ALPHA, PDS_C_BETA, PDS_C_GAMMA),
             'eq_recurrent': eq_rec,
             'eq_unrolling': eq_unr,
             'gradi': {n: {'sigma': s['sigma'], 'max': s['max']}
                       for n, s in stat.items()},
             'err_gradi': err_gradi},
            open(OUT_PKL, 'wb'))


# ---------------- grafico curve GMSE ----------------
it_pds = range(1, len(curva_pds) + 1)
it_unr = range(1, T + 1)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(it_pds, curva_pds,   '--', color='0.6', label='PDS (tunato @20 iter)')
ax.plot(it_pds, curva_pds_c, '-',  color='0.3', label='PDS (tunato a convergenza)')
ax.plot(it_unr, curva_rec, 'o-', ms=3, label='Recurrent Unrolling')
ax.plot(it_unr, curva_unr, 's-', ms=3, label='Unrolling')

if eq_unr:
    ax.axvline(eq_unr, color='k', lw=.8, ls=':')
    ax.annotate(f'{eq_unr} iter', (eq_unr, 0.06), fontsize=8)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(0.05, 2)
ax.set_xlabel('iterazioni / layer')
ax.set_ylabel('GMSE (test)')
ax.grid(alpha=.3, which='both')
ax.legend()
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)


# ---------------- grafico distribuzione dei gradi ----------------
kmax = max(s['gradi'].max() for s in stat.values())
bins = np.arange(0, kmax + 2) - 0.5
centri = np.arange(0, kmax + 1)

stili = {'groundtruth': dict(color='k', lw=2),
         'PDS':         dict(color='0.5', ls='--'),
         'Recurrent':   dict(color='tab:blue', ls='-.'),
         'Unrolling':   dict(color='tab:orange', ls='-')}

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
for nome, s in stat.items():
    h, _ = np.histogram(s['gradi'], bins=bins, density=True)
    ax2.plot(centri, h, label=nome, **stili[nome])

ax2.set_xlabel('grado $k$')
ax2.set_ylabel('frequenza relativa')
ax2.set_xlim(0, kmax)
ax2.grid(alpha=.3)
ax2.legend()
fig2.tight_layout()
fig2.savefig(OUT_PNG_DEG, dpi=150)

print(f'\nsalvati: {OUT_PKL}, {OUT_PNG}, {OUT_PNG_DEG}')

layer = np.arange(1, T + 1)
 
fig3, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
 
for ax, chiave, simbolo in zip(axes,
                               ['alpha', 'beta', 'gamma'],
                               [r'$\alpha^{(t)}$', r'$\beta^{(t)}$', r'$\gamma^{(t)}$']):
    ax.plot(layer, par_unr[chiave], 's-', ms=4,
            color='tab:orange', label='Unrolling')
    # il Recurrent ha un solo valore condiviso: linea orizzontale di riferimento
    ax.axhline(par_rec[chiave][0], color='tab:blue', ls='-.', lw=1,
               label='Recurrent (condiviso)')
    ax.set_ylabel(simbolo)
    ax.grid(alpha=.3)
 
axes[2].set_yscale('log')          # gamma varia di ordini di grandezza
axes[2].set_xlabel('indice del layer $t$')
axes[0].legend(fontsize=8)
axes[0].set_xticks(layer[::2])
fig3.tight_layout()
fig3.savefig(OUT_PNG_PAR, dpi=150)
 
 
# ---------------- profondità effettiva ----------------
# un layer è considerato inattivo se il suo passo è trascurabile
soglia_gamma = 1e-3
attivi = np.where(par_unr['gamma'] > soglia_gamma)[0]
profondita = int(attivi.max()) + 1 if len(attivi) else 0
print(f'\nlayer con gamma > {soglia_gamma:g}: {profondita} su {T}')
print(f'GMSE al layer {profondita}: {curva_unr[profondita-1]:.4f}  '
      f'(finale: {curva_unr[-1]:.4f})')
 