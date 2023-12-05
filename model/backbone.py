# https://github.com/lucidrains/vit-pytorch/blob/main/norm_vit_pytorch/vit.py
from einops import rearrange, repeat
from timm.models.layers import DropPath
import torch
from torch import nn
import torch.nn.functional as F

from .glu import GLU
from .helpers import (
    FFN,
    GLU,
    Lrpe,
    SimpleRMSNorm,
    get_activation_fn,
    get_norm_fn,
    print_params,
)
from .lrpe import Lrpe


##### no cls
class LinearAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        dim_head=64, 
        gate_dim=16,
        dropout=0., 
        num_row_patches=7, # number of patches in a row
        norm_type="layernorm",
        act_fun="swish",
        bias=False,
        use_lrpe=True,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        inner_dim = dim_head * num_heads

        self.num_heads = num_heads
        self.head_dim = dim_head
        self.scale = dim_head ** -0.5

        self.num_row_patches = num_row_patches

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        
        self.output_gate_proj = nn.Sequential(
            nn.Linear(dim, gate_dim, bias=bias),
            nn.Linear(gate_dim, dim, bias=bias),
        )

        self.to_out = nn.Linear(inner_dim, dim)

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(
                dim=dim,
                num_heads=self.num_heads,
            )
        self.layer_norm = nn.LayerNorm(inner_dim)
        self.act_fun = get_activation_fn(act_fun)

    def forward(self, x):
        qkv = self.act_fun(self.to_qkv(x)).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        
        if self.use_lrpe:
            q = self.lrpe(q)
            k = self.lrpe(k)
            
        scale = q.shape[-1] ** 0.5

        # compute
        kv = torch.matmul(k.transpose(-1, -2) / scale, v)
        output = torch.matmul(q / scale, kv)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.layer_norm(output)
        
        # output gate
        output_gate = F.sigmoid(self.output_gate_proj(x))
        output = output * output_gate
        
        # output projection
        output = self.to_out(output)

        return output

class Block(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        dim_head,
        glu_dim, 
        gate_dim=16,
        dropout=0., 
        drop_path=0., 
        num_row_patches=7, # number of patches in a row
        norm_type="layernorm",
        act_fun="swish",
        bias=False,
        use_lrpe=True,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.token_mixer = LinearAttention(
            dim=dim, 
            num_heads=num_heads, 
            dim_head=dim_head, 
            gate_dim=gate_dim,
            dropout=dropout, 
            num_row_patches=num_row_patches, # number of patches in a row
            norm_type=norm_type,
            act_fun=act_fun,
            bias=bias,
            use_lrpe=use_lrpe,
        )
        self.feature_mixer = GLU(dim, glu_dim, act_fun, bias)
        self.token_norm = get_norm_fn(norm_type)(dim)
        self.feature_norm = get_norm_fn(norm_type)(dim)
    
    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.token_norm(x)))
        x = x + self.drop_path(self.feature_mixer(self.feature_norm(x)))

        return x
