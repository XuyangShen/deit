import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

from .helpers import print_module, print_params


class Lrpe(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        # mix lrpe/rope
        self.num_heads = num_heads
        self.theta = nn.Parameter(10000**(-2 / dim * torch.arange(dim // 4)).reshape(num_heads, -1), requires_grad=False)
        self.index = torch.empty(0)
        self.cos = torch.empty(0)
        self.sin = torch.empty(0)

    def extra_repr(self):
        return print_module(self)
    
    def forward(self, x):
        # b, l, d = x.shape
        d = x.shape[-1]
        assert d >= 3
        # split
        e = d // 2
        # 转换为偶数
        if e % 2:
            e += 1
        n = x.shape[-2]
        d = x.shape[-1]
        # last e feature
        x1 = x[..., e:]
        # only operate rope on first e feature
        x = x[..., :e]
        # h, 1, d, 2
        if self.index.shape[0] == 0:
            self.index = torch.arange(n).reshape(1, -1, 1).to(x)
            r = int(n ** 0.5)
            col = self.index // r
            row = self.index % r
            index = col + row

            theta = torch.stack([index * self.theta.unsqueeze(1), index * self.theta.unsqueeze(1)], dim=-1).reshape(self.num_heads, self.index.shape[1], -1)
            self.sin = torch.sin(theta)
            self.cos = torch.cos(theta)
        # (-q1, -q3), (q0, q2) -> (-q1, q0, -q3, q2)
        x_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x = x * self.cos + x_half * self.sin
        
        x_transform = torch.cat([x, x1], dim=-1)

        return x_transform