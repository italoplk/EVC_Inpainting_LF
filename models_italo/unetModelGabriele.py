import torch
import torch.nn as nn
# from einops.layers.torch import Reduce
import sys




class UNetLike(nn.Module):
    def __init__(self, encoder, decoder, compose=lambda x,y: torch.cat([x,y], dim=1)):
        super().__init__()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.compose = compose
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def save(self, path):
        torch.save(self.state_dict(), path)
    def forward(self, X):
        paths = []
        for enc in self.encoder:
            # print("enc", X.shape)
            X = enc(X)
            paths.append(X)
        # print("enc2", X.shape)

        paths = paths[::-1][2:]
        # for p in paths:
            # print(p.shape)
        # print('-----\n')
        X = self.decoder[0](X)
        # print(f'dec {0}: {X.shape}')
        for i, (s, dec) in enumerate(zip(paths, self.decoder[1:])):
            # print(f'dec {i + 1}: {X.shape} {s.shape}')
            X = self.compose(X, s)
            X = dec(X)
        #     print(f'output {X.shape}\n')
        # print(X)

        #     print("shape", s)
        # X = self.decoder[-1](X)
        return X
