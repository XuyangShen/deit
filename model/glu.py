import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import get_activation_fn, print_params


class GLU(nn.Module):
    def __init__(self, d1, d2, act_fun="swish", bias=False):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.l1 = nn.Linear(d1, d2, bias=bias)
        self.l2 = nn.Linear(d1, d2, bias=bias)
        self.l3 = nn.Linear(d2, d1, bias=bias)
        self.act_fun = get_activation_fn(act_fun)

    def forward(self, x):
        o1 = self.l1(x)
        weight = self.act_fun(o1)
        o2 = self.l2(x)
        output = weight * o2
        output = self.l3(output)

        return output
