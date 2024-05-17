from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class GRAMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, degrees = 3, act=nn.SiLU()):
        super(GRAMLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degrees = degrees

        self.act = act

        self.norm = nn.LayerNorm(out_channels, dtype=torch.float32)

        self.beta_weights = nn.Parameter(torch.zeros(degrees + 1, dtype=torch.float32))

        self.grams_basis_weights = nn.Parameter(
            torch.zeros(in_channels, out_channels, degrees + 1, dtype=torch.float32)
        )

        self.base_weights = nn.Parameter(
            torch.zeros(out_channels, in_channels, dtype=torch.float32)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (self.in_channels * (self.degrees + 1.0)),
        )

        nn.init.xavier_uniform_(self.grams_basis_weights)

        nn.init.xavier_uniform_(self.base_weights)

    def beta(self, n, m):
        return (
            ((m + n) * (m - n) * n**2) / (m**2 / (4.0 * n**2 - 1.0))
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
    def gram_poly(self, x, degree):
        p0 = x.new_ones(x.size())

        if degree == 0:
            return p0.unsqueeze(-1)

        p1 = x
        grams_basis = [p0, p1]

        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, i) * p0
            grams_basis.append(p2)
            p0, p1 = p1, p2

        return torch.stack(grams_basis, dim=-1)

    def forward(self, x):

        basis = F.linear(self.act(x), self.base_weights)

        x = torch.tanh(x).contiguous()

        grams_basis = self.act(self.gram_poly(x, self.degrees))

        y = einsum(
            grams_basis,
            self.grams_basis_weights,
            "b l d, l o d -> b o",
        )

        y = self.act(self.norm(y + basis))

        y = y.view(-1, self.out_channels)
        
        return y
