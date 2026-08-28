import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from unrolling import Unrolling
from pds import PDS
from utils import gmse

GRAPH_TYPE = os.environ.get('GRAPH_TYPE', 'ER')
SHARED   = os.environ.get('SHARED', '1') == '1'   # True -> Recurrent Unrolling
DATA     = f'generated_data/data_{GRAPH_TYPE}50_train.pkl'
DATA_PLOT = f'data_for_plot/{GRAPH_TYPE}'
os.makedirs(DATA_PLOT, exist_ok=True)
CKPT_DIR = f'trained_models/{GRAPH_TYPE}'
os.makedirs(CKPT_DIR, exist_ok=True)
N, T     = 50, 3
EPOCHS   = 120
N_TR, N_VA = 8000, 2000
TAG      = f"{'recurrent' if SHARED else 'unrolling'}_{GRAPH_TYPE}50_T3"
SEED     = 0

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev)

torch.manual_seed(SEED)

d = pickle.load(open(DATA, 'rb'))
y = torch.tensor(d['y'], dtype=torch.float32)
w = torch.tensor(d['w'], dtype=torch.float32)
print(f'dataset: {y.shape[0]} campioni, {y.shape[1]} archi')

# Training set, validation set, test set
tr = DataLoader(TensorDataset(y, w), batch_size=32, shuffle=True)

d_val = pickle.load(open(f'generated_data/data_{GRAPH_TYPE}50_val.pkl', 'rb'))
y_val = torch.tensor(d_val['y'], dtype=torch.float32).to(dev)
w_val = torch.tensor(d_val['w'], dtype=torch.float32).to(dev)

d_test = pickle.load(open(f'generated_data/data_{GRAPH_TYPE}50_test.pkl', 'rb'))
y_test = torch.tensor(d_test['y'], dtype=torch.float32).to(dev)
w_test = torch.tensor(d_test['w'], dtype=torch.float32).to(dev)

def loss_fn(outs, w, tau=0.9):
    T = len(outs)
    den = (w**2).sum(1)
    loss = sum(tau**(T-t) * (((o - w)**2).sum(1) / den).mean() for t, o in enumerate(outs))
    return loss
    
Y_np, W_np = d['y'], d['w']
alphas = [0.5, 1, 2, 4, 8, 10]
betas  = [0.5, 1, 2, 4, 8]
gammas = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]

for T in [3, 5, 8, 12, 20]:
    TAG = f"{'recurrent' if SHARED else 'unrolling'}_{GRAPH_TYPE}50_T{T}"
    
    solver = PDS(alpha=2, beta=2, gamma=0.1)
    solver.tune(Y_np, W_np, alphas, betas, gammas, max_iter=T)

    model = Unrolling(N=N, T=T, shared=SHARED, alpha=solver.alpha, beta=solver.beta, gamma=solver.gamma).to(dev)

    opt   = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    
    with torch.no_grad():
        print(f'GMSE val a init (= PDS con gamma0): {gmse(w_val, model(y_val)[-1]):.4f}')

    # --- TRAIN ---
    best = 1e9
    hist = []

    for ep in range(EPOCHS):
        model.train()
        tot = 0.
        for yb, wb in tr:
            yb, wb = yb.to(dev), wb.to(dev)
            opt.zero_grad()
            pred = model(yb)
            loss = loss_fn(pred, wb)
    
            if not torch.isfinite(loss):
                raise RuntimeError(f'loss non finita a epoca {ep}')
    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * yb.shape[0]
        sched.step()
    
        model.eval()
        with torch.no_grad():
            g = gmse(w_val, model(y_val)[-1]) # Calcolo GMSE sull'ultimo layer ([-1])
        hist.append((tot / N_TR, g))
    
        if g < best:
            best = g
            torch.save(model.state_dict(), f'{CKPT_DIR}/{TAG}.pt')
            if T == 20:
                # copia "di riferimento" senza suffisso _T, usata da curves.py (ckpt_rec/ckpt_unr)
                flat_tag = f"{'recurrent' if SHARED else 'unrolling'}_{GRAPH_TYPE}50"
                torch.save(model.state_dict(), f'{CKPT_DIR}/{flat_tag}.pt')
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f'Epoca {ep:3d}  train loss {tot/N_TR:8.4f}   val GMSE {g:.4f}'
                  + ('  *' if g == best else ''))
    
    # --- TEST ---
    model.load_state_dict(torch.load(f'{CKPT_DIR}/{TAG}.pt', map_location=dev))
    model.eval()
    
    with torch.no_grad():
        outs = model(y_test)
        print(f'\nTEST GMSE: {gmse(w_test, outs[-1]):.4f}')
        print('GMSE per layer:', [round(gmse(w_test, o), 4) for o in outs])
    
        gamma_np = model.gamma.detach().cpu().numpy()
        alpha_np = model.alpha.detach().cpu().numpy()
        beta_np  = model.beta.detach().cpu().numpy()
        print('gamma:', gamma_np.round(4))
        print('alpha:', alpha_np.round(4))
        print('beta :', beta_np.round(4))
    
    pickle.dump({'hist': hist,
                 'gmse_layer': [gmse(w_test, o) for o in outs],
                 'alpha': alpha_np,
                 'beta':  beta_np,
                 'gamma': gamma_np},
                open(f'{DATA_PLOT}/results_{TAG}.pkl', 'wb'))