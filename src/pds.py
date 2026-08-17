import pickle
import torch
import math
from utils import make_D, prox2, gmse

class PDS():
    def __init__(self, alpha, beta, gamma):
            self.alpha = alpha    # penalita' della log-barrier
            self.beta = beta      # penalita' del termine l2
            self.gamma = gamma    # step size
    
    def solve(self, y, max_iter=20):
        y = torch.as_tensor(y, dtype=torch.float32)
        E = y.shape[0]
        N = int(round((1 + math.sqrt(1 + 8 * E)) / 2))
        D = make_D(N)

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

    def tune(self, Y, W, alphas, betas, gammas, n_samples=100):
        best = [1e9, None]
        W_true = torch.as_tensor(W[:n_samples], dtype=torch.float32)
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    self.alpha, self.beta, self.gamma = alpha, beta, gamma
                    W_pred = torch.stack([self.solve(Y[i]) for i in range(n_samples)])
                    score = gmse(W_true, W_pred)
                    if score < best[0]:
                        best = [score, (alpha, beta, gamma)]
        self.alpha, self.beta, self.gamma = best[1]
        print("Best GMSE: ", best[0], " Best parameters: ", best[1])

    @staticmethod
    def pds_curve(y, alpha, beta, gamma, max_iter, w_true, return_w=False):
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

            curva.append(gmse(w_true, w))
        return w if return_w else curva

if __name__ == '__main__':
    d = pickle.load(open('generated_data/data_BA50_train.pkl', 'rb'))
    Y, W = d['y'], d['w']

    solver = PDS(alpha=2, beta=2, gamma=0.1)
    solver.tune(Y, W, [0.5,1,2,4,8], [0.5,1,2,4,8], [0.01,0.05,0.1,0.3,0.5,1.0])

    d_test = pickle.load(open('generated_data/data_BA50_test.pkl', 'rb'))
    Y_test, W_test = d_test['y'], d_test['w']

    W_true = torch.as_tensor(W_test, dtype=torch.float32)
    W_pred = torch.stack([solver.solve(Y_test[i], max_iter=20) for i in range(len(Y_test))])
    print(f'PDS @20 iter, TEST GMSE: {gmse(W_true, W_pred):.4f}')
    print('parametri:', solver.alpha, solver.beta, solver.gamma)