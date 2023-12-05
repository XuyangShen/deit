# Network architecture under test
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from .model_vit import Vit


##### Deit tiny
@register_model
def norm_linear_vit_tiny(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 6
    gate_dim = 16
    glu_dim = int(dim * 8 / 3)
    dropout = 0.0
    bias = False
    act_fun = "swish"
    use_lrpe = True
    model = Vit(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        glu_dim=glu_dim,
        gate_dim=gate_dim,
        act_fun=act_fun,
        bias=bias,
        use_lrpe=use_lrpe,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def norm_linear_vit_tiny_abs(pretrained=False, **kwargs):
    patch_size = 16
    dim = 192
    depth = 12
    num_heads = 6
    gate_dim = 16
    glu_dim = int(dim * 8 / 3)
    dropout = 0.0
    bias = False
    act_fun = "swish"
    use_lrpe = False
    use_pos = True
    model = Vit(
        patch_size=patch_size, 
        dim=dim, 
        depth=depth, 
        num_heads=num_heads, 
        glu_dim=glu_dim,
        gate_dim=gate_dim,
        act_fun=act_fun,
        bias=bias,
        use_lrpe=use_lrpe,
        use_pos=use_pos,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model