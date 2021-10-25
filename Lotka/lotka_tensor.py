import numpy as np

import torch
from torch import nn

from ptflops import get_model_complexity_info

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.variable = nn.Parameter((torch.randn(())), requires_grad=True)
    def forward(self, y):
        dy = torch.empty(y.shape)
        dy[:, 0] =  y[:, 0] - y[:, 0] * y[:, 1]
        dy[:, 1] = -y[:, 1] + y[:, 0] * y[:, 1]
        return dy

if __name__ == '__main__':
    func = ODEFunc()
    macs, params = get_model_complexity_info(func, (2, ), as_strings=False)
    print(macs)