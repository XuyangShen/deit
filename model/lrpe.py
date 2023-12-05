import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

from .helpers import print_module, print_params


class Lrpe(nn.Module):
    def __init__(self, dim, num_heads=8, dims=[1]):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.dims = dims

        # mix lrpe/rope
        self.num_heads = num_heads
        self.theta = nn.Parameter(10000**(-2 / dim * torch.arange(dim // 4)).reshape(num_heads, -1), requires_grad=False)
        self.index = torch.empty(0)

    def extra_repr(self):
        return print_module(self)

    def forward(self, x):
        # b, l, e
        for dim in self.dims:
            x = self.mix_rope(x, dim)
        return x

    def mix_rope(self, x, op_dim=1):
        # b, l, d = x.shape
        d = x.shape[-1]
        assert d >= 3
        # split
        e = d // 2
        # 转换为偶数
        if e % 2:
            e += 1
        return self.mix_transform(x, e, op_dim)

    def mix_transform(self, x, e, op_dim):
        assert e % 2 == 0
        assert op_dim in [-2, -3]
        l = x.shape[op_dim]
        d = x.shape[-1]
        m = len(x.shape)
        # last e feature
        x1 = x[..., e:]
        # only operate rope on first e feature
        x = x[..., :e]
        # h, 1, 1, d, 2
        theta = torch.stack([self.theta, self.theta], dim=-1).reshape(self.num_heads, -1).unsqueeze(1).unsqueeze(1)
        ### l
        if self.index.shape[0] < l:
            self.index = torch.arange(l).to(x)
        index = self.index[:l]
        
        if op_dim == -2:
            # n, 1
            index = index.unsqueeze(-1)
        elif op_dim == -3:
            # n, 1, 1
            index = index.unsqueeze(-1).unsqueeze(-1)
        theta = theta * index

        # (-q1, -q3), (q0, q2) -> (-q1, q0, -q3, q2)
        x_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x_transform = x * torch.cos(theta) + x_half * torch.sin(theta)
        x_transform = torch.cat([x_transform, x1], dim=-1)

        return x_transform