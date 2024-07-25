from functools import lru_cache
import torch
import torch.nn as nn

class GRAMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, degrees = 3):
        super(GRAMLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degrees = degrees

        self.beta_weights = nn.Parameter(torch.zeros(degrees + 1, dtype=torch.float32))
        self.basis_weights = nn.Parameter(
            torch.zeros(in_channels, out_channels, degrees + 1, dtype=torch.float32)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (self.in_channels * (self.degrees + 1.0)),
        )

        nn.init.xavier_uniform_(self.basis_weights)

    def beta(self, n, m):
        return (
            ((m + n) * (m - n) * n**2) / (m**2 / (4.0 * n**2 - 1.0))
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
    def get_basis(self, x, degree):
        p0 = x.new_ones(x.size())
        if degree == 0:
            return p0.unsqueeze(-1)
        p1 = x
        basis = [p0, p1]
        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, i) * p0
            basis.append(p2)
            p0, p1 = p1, p2
        return torch.stack(basis, dim=-1)

    def forward(self, x):
        x = torch.tanh(x).contiguous()
        basis = self.get_basis(x, self.degrees)
        y = torch.einsum(
            "b l d, l o d -> b o",
            basis,
            self.basis_weights            
        )
        y = y.view(-1, self.out_channels)
        return y

#Example that works for MNIST
class GRAM(nn.Module):
    def __init__(self):
        super(GRAM, self).__init__()
        self.layers = nn.Sequential(
            GRAMLayer(28 * 28, 32, 4), 
            nn.LayerNorm(32),
            GRAMLayer(32, 16, 4), 
            nn.LayerNorm(16),
            GRAMLayer(16, 10, 4)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)
