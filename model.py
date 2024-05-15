from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class GRAMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, degrees=3, act=nn.SiLU(), bias=False):
        super(GRAMLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degrees = degrees
        self.bias = bias

        self.beta_weights = nn.Parameter(
            torch.zeros(1, dtype=torch.float32),
            requires_grad=True
        )

        self.grams_weights = nn.Parameter(
            torch.zeros(in_channels, out_channels, degrees + 1)
        )

        self.base_weights = nn.Parameter(
            torch.zeros(out_channels, in_channels),
            requires_grad=True
        )

        self.act = act

        self.norm = nn.LayerNorm(out_channels)

        nn.init.kaiming_uniform_(self.grams_weights, mode="fan_out", nonlinearity="linear")
        nn.init.kaiming_uniform_(self.base_weights, mode="fan_out", nonlinearity="linear")
        nn.init.constant_(self.beta_weights, 1.0)
        
    def beta(self, n, m):
        return ((m + n) * (m - n) * n**2) / (m**2 / (4.0 * n**2 - 1.0))

    @lru_cache(maxsize=128)
    def gram_poly(self, x, degree):
        p0 = x.new_ones(x.size())
        if degree == 0:
            return p0.unsqueeze(-1)
        p1 = x
        grams = [p0, p1]
        for i in range(2, degree + 1):
            p2 = x * p1 - (self.beta(i - 1, i) * self.beta_weights) * p0
            grams.append(p2)
            p0, p1 = p1, p2

        return torch.stack(grams, dim=-1)

    def forward(self, x):
        
        basis = F.linear(self.act(x), self.base_weights)

        x = torch.tanh(x).contiguous()

        grams = self.gram_poly(x, self.degrees)

        y = einsum(
            grams,
            self.grams_weights,
            "b l d, l o d -> b o",
        )

        y = self.act(self.norm(y + basis))

        return y