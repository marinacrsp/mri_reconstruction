import math
import numpy as np
import torch
from torch import nn


class Siren(nn.Module):
    def __init__(
        self,
        coord_dim=3,
        latent_dim=64,
        hidden_dim=512,
        n_layers=4,
        out_dim=2,
        omega_0=30,
        L = 10,
        input_embd_layer = 2,

    ) -> None:
        super().__init__()
        self.L = L
        self.n_layers = n_layers
        self.input_embd_layer = input_embd_layer

        # Precompute the scaling factors for the coordinate encoding.
        L_mult = torch.pow(2, torch.arange(self.L)) * math.pi 
        self.register_buffer("L_mult", L_mult) 
        coord_encoding_dim = self.L * 2 * coord_dim # 20 * 3

        # Input
        self.sine_layers = nn.ModuleList()
                
        self.sine_layers.append(   
                SineLayer(
                coord_encoding_dim + latent_dim,
                hidden_dim,
                is_first=True,
                omega_0=omega_0,
            ))

        count = 1       
        for layer_idx in range(n_layers - 1):
            
            block = nn.ModuleList()
            if count <= 2 and (layer_idx + 1) % 2 == 0: # Append PE
                count += 1
                block.append(
                    SineLayer(
                        hidden_dim + latent_dim + coord_encoding_dim,
                        hidden_dim,
                        is_first=False,
                        omega_0=omega_0)
                    )
                
                block.append(nn.LayerNorm(hidden_dim))
                
            else: 
                block.append(
                    SineLayer(
                        hidden_dim,
                        hidden_dim,
                        is_first=False,
                        omega_0=omega_0,
                    ))
                
                block.append(nn.LayerNorm(hidden_dim))
                
            self.sine_layers.append(block)
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        with torch.no_grad():
            self.output_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_0, np.sqrt(6 / hidden_dim) / omega_0
            )

        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, coords, pe_levels):
        # Positional encodings.
        x0 = coords.unsqueeze(-1) * self.L_mult
        x0 = torch.cat([torch.sin(x0), torch.cos(x0)], dim=-1)
        x0 = x0.view(x0.size(0), -1)
        x = x0.clone()
        # Concatenate embeddings and positional encodings.
        level_idx = 0
        for layer_idx, layer in enumerate(self.sine_layers):

            if level_idx <= 2 and layer_idx % 2 == 0: # For now it works with 3 PE embeddings
                if layer_idx != 0:
                    x = torch.cat([x, x0, pe_levels[level_idx]], dim=-1)
                    level_idx += 1
                else:
                    x = torch.cat([x, pe_levels[level_idx]], dim=-1)
                    level_idx += 1
                    
            # Then iterate through the blocks (Sine layer & layer normalization)
            if layer_idx != 0:
                for _, module in enumerate(layer):
                    x = module(x)
            else:
                x = layer(x)

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
