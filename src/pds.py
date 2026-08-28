import pickle
import torch
import math
from utils import make_D, prox2, gmse

GRAPH_TYPE = 'ER'
TRAIN = f'generated_data/data_{GRAPH_TYPE}50_train.pkl'

class PDS():
    def __init__(self, alpha, beta, gamma):
            self.alpha = alpha    # penalita' della log-barrier
            self.beta = beta      # penalita' del termine l2
            self.gamma = gamma    # step size
    
    def solve(self, y, max_iter=500):
        """"
        PDS in forma singola su y di shape (E,).
        Restituisce la stima w di shape (E,).
        """
        y = torch.as_tensor(y, dtype=torch.float32)
        E = y.shape[0]
        N = int(round((1 + math.sqrt(1 + 8 * E)) / 2))
        if not hasattr(self, '_D') or self._D.shape[0] != N:
            self._D = make_D(N)
        D = self._D

        w = torch.zeros(E)
        v = torch.zeros(N)

        for _ in range(max_iter):
            r1 = w - self.gamma * (2 * self.beta * w + 2 * y + D.T @ v)
            r2 = v + self.gamma * (D @ w)

            p1 = torch.clamp(r1, min=0.0)
            p2 = prox2(r2, self.alpha, self.gamma)

            q1 = p1 - self.gamma * (2 * self.beta * p1 + 2 * y + D.T @ p2)
            q2 = p2 + self.gamma * (D @ p1)

            w = w - r1 + q1
            v = v - r2 + q2

        return w

    def tune(self, Y, W, alphas, betas, gammas, n_samples=100, max_iter=20):
        """
        Tuning dei parametri alpha, beta, gamma su un sottoinsieme di n_samples
        """
        best = [1e9, None]
        W_true = torch.as_tensor(W[:n_samples], dtype=torch.float32)
        tot = len(alphas) * len(betas) * len(gammas)
        i = 0

        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    i += 1
                    self.alpha, self.beta, self.gamma = alpha, beta, gamma
                    W_pred = torch.stack([self.solve(Y[k], max_iter=max_iter) for k in range(n_samples)])
                    score = gmse(W_true, W_pred)
                    if score < best[0]:
                        best = [score, (alpha, beta, gamma)]
                        print(f'  [{i}/{tot}] nuovo best: {score:.4f}  'f'(a={alpha}, b={beta}, g={gamma})', flush=True)
                    elif i % 25 == 0:
                        print(f'  [{i}/{tot}]...', flush=True)
                        
        self.alpha, self.beta, self.gamma = best[1]
        print(f'Best GMSE: {best[0]:.4f}  Best parameters: {best[1]}', flush=True)

    @staticmethod
    def pds_curve(y, alpha, beta, gamma, max_iter, w_true, return_w=False):
        """
        PDS in forma batched su y di shape (B, E).
        Se return_w=False restituisce la lista dei GMSE dopo ogni iterazione,
        altrimenti la stima finale w di shape (B, E).
        """
        E = y.shape[1]
        n = int(round((1 + math.sqrt(1 + 8 * E)) / 2))
        D = make_D(n).to(y.device)

        w = torch.zeros_like(y)
        v = torch.zeros(y.shape[0], D.shape[0], device=y.device)

        curva = []
        for _ in range(max_iter):
            r1 = w - gamma * (2 * beta * w + 2 * y + v @ D)
            r2 = v + gamma * (w @ D.T)

            p1 = torch.clamp(r1, min=0.)
            p2 = prox2(r2, alpha, gamma)

            q1 = p1 - gamma * (2 * beta * p1 + 2 * y + p2 @ D)
            q2 = p2 + gamma * (p1 @ D.T)

            w = w - r1 + q1
            v = v - r2 + q2

            if not return_w:
                curva.append(gmse(w_true, w))

        return w if return_w else curva

if __name__ == '__main__':
    d = pickle.load(open(TRAIN, 'rb'))
    Y, W = d['y'], d['w']
    
    solver = PDS(alpha=2, beta=2, gamma=0.1)
    solver.tune(Y, W, [0.5, 1, 2, 4, 8], [0.5, 1, 2, 4, 8], [0.01, 0.05, 0.1, 0.3])
    
    i = 0
    wh = solver.solve(Y[i], max_iter=5000)
    w_true = torch.as_tensor(W[i], dtype=torch.float32)
    
    print('archi veri:', (w_true > 1e-4).sum().item(), ' archi stimati:', (wh > 1e-4).sum().item())
    print('sum w:', w_true.sum().item(), ' sum ŵ:', wh.sum().item())
    print('gmse:', (((wh - w_true) ** 2).sum() / (w_true ** 2).sum()).item())
