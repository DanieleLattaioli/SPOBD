import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from unrolling import Unrolling
from utils import gmse

TRAIN    = 'generated_data/data_BA50_train.pkl'
VAL      = 'generated_data/data_BA50_val.pkl'
TEST     = 'generated_data/data_BA50_test.pkl'
N, T     = 50, 20
ALFA0    = 2.0
BETA0    = 2.0
GAMMA0   = 0.1                        
SHARED   = False                       # True -> Recurrent Unrolling
EPOCHS   = 200
N_TR, N_VA = 8000, 2000
TAG      = f"{'recurrent' if SHARED else 'unrolling'}_BA50"
SEED     = 0

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev)

torch.manual_seed(SEED)

# Caricamento dati
d = pickle.load(open(TRAIN, 'rb'))
y = torch.tensor(d['y'], dtype=torch.float32)
w = torch.tensor(d['w'], dtype=torch.float32)

tr = DataLoader(TensorDataset(y, w), batch_size=32, shuffle=True)

d_val = pickle.load(open(VAL, 'rb'))
y_val = torch.tensor(d_val['y'], dtype=torch.float32).to(dev)
w_val = torch.tensor(d_val['w'], dtype=torch.float32).to(dev)

d_test = pickle.load(open(TEST, 'rb'))
y_test = torch.tensor(d_test['y'], dtype=torch.float32).to(dev)
w_test = torch.tensor(d_test['w'], dtype=torch.float32).to(dev)

# Funzione di loss
def loss_fn(outs, w, tau=0.9):
    T = len(outs)
    den = (w**2).sum(1)
    loss = sum(tau**(T-t) * (((o - w)**2).sum(1) / den).mean() for t, o in enumerate(outs))
    return loss

# Definizione del modello e ottimizzatore
model = Unrolling(N=N, T=T, shared=SHARED, alpha=ALFA0, beta=BETA0, gamma=GAMMA0).to(dev)
opt   = torch.optim.Adam(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

with torch.no_grad():
    print(f'GMSE val a init (= PDS con gamma0): {gmse(w_val, model(y_val)[-1]):.4f}')

# Training loop
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
            raise RuntimeError(f'loss non finita a epoca {ep}: abbassa gamma0 o lr')

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
        torch.save(model.state_dict(), f'trained_models/{TAG}.pt')
    if ep % 5 == 0 or ep == EPOCHS - 1:
        print(f'Epoca {ep:3d}  train loss {tot/N_TR:8.4f}   val GMSE {g:.4f}'
              + ('  *' if g == best else ''))

# Test
model.load_state_dict(torch.load(f'trained_models/{TAG}.pt', map_location=dev))
model.eval()
with torch.no_grad():
    outs = model(y_test)
    print(f'\nTEST GMSE: {gmse(w_test, outs[-1]):.4f}   (PDS: 0.1283)')
    print('GMSE per layer:', [round(gmse(w_test, o), 4) for o in outs])

    gamma_np = model.gamma.detach().cpu().numpy()
    alpha_np = model.alpha.detach().cpu().numpy()
    beta_np  = model.beta.detach().cpu().numpy()
    print('gamma:', gamma_np.round(4))
    print('alpha:', alpha_np.round(4))
    print('beta :', beta_np.round(4))
