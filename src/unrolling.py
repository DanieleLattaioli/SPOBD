import math
import torch, torch.nn as nn
import torch.nn.functional as F
from utils import make_D, prox2


def inv_softplus(x):
    return math.log(math.exp(x) - 1)

class Unrolling(nn.Module):
    def __init__(self, N, T=20, shared=False, alpha=2, beta=2, gamma=0.1):
        super().__init__()
        self.T = T
        self.D = make_D(N)
        n = 1 if shared else T
        self.shared = shared

        self._alpha = nn.Parameter(torch.full((n,), inv_softplus(float(alpha))))
        self._beta  = nn.Parameter(torch.full((n,), inv_softplus(float(beta))))
        self._gamma = nn.Parameter(torch.full((n,), inv_softplus(float(gamma))))

    @property
    def alpha(self):
        return F.softplus(self._alpha)

    @property
    def beta(self):
        return F.softplus(self._beta)

    @property
    def gamma(self):
        return F.softplus(self._gamma)

    def forward(self, y):

        D = self.D.to(y.device)

        w = torch.zeros_like(y)
        v = torch.zeros(y.shape[0], D.shape[0], device=y.device)

        out = []

        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        for t in range(self.T):
            k = 0 if self.shared else t

            a, b, g = alpha[k], beta[k], gamma[k]

            r1 = w - g * (2*b*w + 2 * y + v @ D)
            r2 = v + g * (w @ D.T)

            p1 = torch.clamp(r1, min=0.)
            p2 = prox2(r2, a, g)

            q1 = p1 - g * (2 * b * p1 + 2 * y + p2 @ D)
            q2 = p2 + g * (p1 @ D.T)

            w = w - r1 + q1
            v = v - r2 + q2

            out.append(w)
        return out