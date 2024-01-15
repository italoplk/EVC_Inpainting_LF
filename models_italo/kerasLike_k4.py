import os.path
from argparse import Namespace

import torch
import torch.nn as nn
from models_italo.unetModelGabriele import UNetLike
from models_italo.ModelGabriele import RegModel
from torch.nn import Conv2d, ConvTranspose2d


class UNetSpace(nn.Module):
    def __init__(self, num_filters, no_skip):
        super().__init__()

        n_filters = num_filters
        print("n_filters: ", n_filters)

        if no_skip:
            type_mode = RegModel
            mul_fact = 1
            print("kernels 4 no-skip")

        else:
            type_mode = UNetLike
            mul_fact = 2
            print("kernels 4 skip")

        model = type_mode([  # 18, 64²
            nn.Sequential(
                Conv2d(1, n_filters, 3, stride=1, padding=1), nn.PReLU(),  # 10, 64²
                Conv2d(n_filters, n_filters, 3, stride=2, padding=1), nn.PReLU(),  # 10, 32²
            ),
            nn.Sequential(
                Conv2d(n_filters, (n_filters * 2), 3, stride=1, padding=1), nn.PReLU(),  # 10, 32²
                Conv2d((n_filters*2), (n_filters*2), 3, stride=2, padding=1), nn.PReLU(),  # 10, 16²
            ),
            nn.Sequential(
                Conv2d((n_filters*2), (n_filters*4), 3, stride=1, padding=1), nn.PReLU(),  # 10, 16²
                Conv2d((n_filters*4), (n_filters*4), 3, stride=2, padding=1), nn.PReLU(),  # 10, 8²
            ),
            nn.Sequential(
                Conv2d((n_filters*4), (n_filters*8), 3, stride=1, padding=1), nn.PReLU(),  # 10, 8²
                Conv2d((n_filters*8), (n_filters*8), 3, stride=2, padding=1), nn.PReLU(),  # 10, 4²
            ),
            nn.Sequential(
                Conv2d((n_filters*8), 512, 3, stride=1, padding=1), nn.PReLU(),  # 10, 4
            ),

        ], [
            nn.Sequential(  # 10, 4
                nn.ConvTranspose2d(512, n_filters*4, kernel_size=4, stride=2, padding=1), nn.PReLU()
            ),
            nn.Sequential(  # 10, 8
                nn.ConvTranspose2d(mul_fact*(n_filters*4), n_filters * 2, kernel_size=4, stride=2, padding=1), nn.PReLU()
            ),
            nn.Sequential(  # 10, 510²
                nn.ConvTranspose2d(mul_fact * (n_filters * 2), n_filters, kernel_size=4, stride=2, padding=1), nn.PReLU()
            ),
            nn.Sequential(  # 10, 510²a
                nn.ConvTranspose2d(mul_fact * (n_filters), 1, 4, stride=2, padding=1),
                nn.Sigmoid()
            )

        ]) # compose=lambda batch_size, y: batch_size+y)
        self.network = model
        # self.name = name + '.data'
        # try:
        #     if os.path.exists(self.name):
        #         self.load_state_dict(torch.load(self.name))
        # except RuntimeError:
        #     pass
    
    def save(self):
        torch.save(self.state_dict(), self.name)

    def forward(self, X):
        #assert(tuple(X.shape[1:]) == (1,8*64,8*64))
        return self.network(X)


#
# params = Namespace()
# dims = (8,1,64,64)
# dims_out = (8,1,32,32)
# (params.num_views_ver, params.num_views_hor, params.predictor_size, params.predictor_size) = dims
# params.num_filters = 32
#
# params.no_skip = True
# # print(params)
# model = UNetSpace("unet_space", params)
# model.eval()
# zeros = torch.zeros(1, 1, 64, 64)
# zeros_t = torch.zeros(8, 1, 32, 32)
# lossf = nn.MSELoss()
#
# from torchsummary import summary
# with torch.no_grad():
#     batch_size = model(zeros)
#     # print("batch_size: ", batch_size.shape)
#     # batch_size = batch_size[:,:,-32:, -32:]

    # summary(model, (1, 64, 64), depth=100)
    # print(batch_size.shape)
    # batch_size = batch_size[:, :, -32:, -32:]
    # print(batch_size.shape)

    # print(lossf(zeros_t, batch_size))
