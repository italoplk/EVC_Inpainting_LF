import torch
import torch.nn as nn
from einops.layers.torch import Reduce
import sys

#MODEL WITH NO SKIP

class RegModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def save(self, path):
        torch.save(self.state_dict(), path)
    def forward(self, X):
        for i,enc in enumerate(self.encoder):
            # print(f'enc {i}: {X.shape}')
            X = enc(X)
            # print(f'output {X.shape}\n')
        # print('-----')
        # print("enc2", X.shape)

        for i, (dec) in enumerate(self.decoder):
            # print(f'dec {i}: {X.shape}')
            X = dec(X)
            # print(f'output {X.shape}\n')
        # print(X)

        #     print("shape", s)
        # X = self.decoder[-1](X)
        return X
