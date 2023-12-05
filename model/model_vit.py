# https://github.com/lucidrains/vit-pytorch/blob/main/norm_vit_pytorch/vit.py
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch
from torch import nn
import torch.nn.functional as F

from .backbone import Block
from .helpers import (
    FFN,
    GLU,
    Lrpe,
    SimpleRMSNorm,
    get_activation_fn,
    get_norm_fn,
    get_patch_embedding,
    pair,
    print_module,
    print_params,
)


##### no cls
class Vit(nn.Module):
    def __init__(
        self, 
        *, 
        img_size=224, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        num_heads, 
        glu_dim, 
        gate_dim=16,
        channels=3, 
        drop_rate=0., 
        emb_dropout=0., 
        drop_path_rate=0.,
        use_pos=False,
        # add
        norm_type="layernorm",
        act_fun="1+elu",
        bias=False,
        use_lrpe=True,
        drop_block_rate=None
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        
        patch_dim = channels * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
 
        self.use_pos = use_pos
        num_row_patches = image_height // patch_height
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        if self.use_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([])
        dim_head = dim // num_heads
        for i in range(depth):
            self.layers.append(
                Block(
                    dim=dim, 
                    num_heads=num_heads, 
                    dim_head=dim_head,
                    glu_dim=glu_dim,
                    gate_dim=gate_dim,
                    dropout=drop_rate, 
                    drop_path=drop_path_rate,
                    num_row_patches=num_row_patches, # number of patches in a row
                    norm_type=norm_type,
                    act_fun=act_fun,
                    bias=bias,
                    use_lrpe=use_lrpe,
                )
            )

        # classification head
        self.head = nn.Sequential(
            get_norm_fn(norm_type)(dim),
            nn.Linear(dim, num_classes)
        )
        
    def extra_repr(self):
        return print_module(self)
        
    def forward_features(self, x):
        B = x.shape[0]
        x = self.to_patch_embedding(x)
        x = rearrange(x, 'b h w d -> b (h w) d')
        b, n, _ = x.shape
        
        if self.use_pos:
            x = x + self.pos_embedding
        x = self.dropout(x)

        for i, block in enumerate(self.layers):
            x = block(x)
        
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x
