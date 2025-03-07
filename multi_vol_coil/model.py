import math
import numpy as np
import torch
from torch import nn


class Siren(nn.Module):
    def __init__(
        self,
        coord_dim=3,
        vol_embedding_dim=256,
        coil_embedding_dim = 64,
        hidden_dim=512,
        n_layers=4,
        out_dim=2,
        omega_0=30,
        L = 10,

    ) -> None:
        super().__init__()
        self.L = L

        # Precompute the scaling factors for the coordinate encoding.
        L_mult = torch.pow(2, torch.arange(self.L)) * math.pi
        self.register_buffer("L_mult", L_mult)
        coord_encoding_dim = self.L * 2 * coord_dim
        breakpoint()
        self.sine_layers = [
            SineLayer(
                coord_encoding_dim + vol_embedding_dim*2 + coil_embedding_dim*2,
                hidden_dim,
                is_first=True,
                omega_0=omega_0,
            )
        ]
        for layer_idx in range(n_layers - 1):
            # We have a residual connection at this layer (hence the different input dimension).
            if layer_idx == n_layers // 2 - 1:
                self.res_layer_idx = layer_idx + 1
                self.sine_layers.append(
                    SineLayer(
                        hidden_dim + vol_embedding_dim*2 + coil_embedding_dim*2,
                        hidden_dim,
                        is_first=False,
                        omega_0=omega_0,
                    )
                )
            else:
                self.sine_layers.append(
                    SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0)
                )
                # self.sine_layers.append(nn.LayerNorm(hidden_dim))
                # self.sine_layers.append(nn.BatchNorm1d(hidden_dim))
        self.sine_layers = nn.ModuleList(self.sine_layers)

        self.output_layer = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            self.output_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_0, np.sqrt(6 / hidden_dim) / omega_0
            )

        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, coords, vol_embedding, coil_embedding):
        # Positional encodings.
        
        x = coords.unsqueeze(-1) * self.L_mult
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(x.size(0), -1)
        
        vol_embedding = torch.cat([torch.cos(vol_embedding), torch.sin(vol_embedding)], dim=-1)
        coil_embedding = torch.cat([torch.cos(coil_embedding), torch.sin(coil_embedding)], dim=-1)

        # Concatenate embeddings and positional encodings.
        x = torch.cat([vol_embedding, coil_embedding, x], dim=-1)

        for layer_idx, layer in enumerate(self.sine_layers):
            # Residual connection.
            if layer_idx == self.res_layer_idx:
                x = torch.cat([vol_embedding, coil_embedding, x], dim=-1)

            x = layer(x)
            # x = self.dropout(x)

        return self.output_layer(x)

class SineLayer(nn.Module):
    """Linear layer with sine activation. Adapted from Siren repo"""

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)


        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
