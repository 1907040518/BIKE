from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AfterReconstruction(nn.Identity):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes

class CrossFramelAttentionBlockOrigin(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath = 0., T=0, ):
        super().__init__()
        self.T = T

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head,)
           
        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x):
        l, bt, d = x.size()
        b = bt // self.T
        x = x.view(l, b, self.T, d) 

        msg_token = self.message_fc(x[0,:,:,:]) 
        msg_token = msg_token.view(b, self.T, 1, d) 
        
        msg_token = msg_token.permute(1,2,0,3).view(self.T, b, d) 
        msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token),self.message_ln(msg_token),self.message_ln(msg_token),need_weights=False)[0])
        msg_token = msg_token.view(self.T, 1, b, d).permute(1,2,0,3)
        
        x = torch.cat([x, msg_token], dim=0)
        
        x = x.view(l+1, -1, d)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x[:l,:,:]
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath = 0., T=0 ):
        super().__init__()
        self.T = T

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head,)
           
        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x, use_checkpoint=False):
        l, bt, d = x.size()
        b = bt // self.T
        x = x.view(l, b, self.T, d) 

        msg_token = self.message_fc(x[0,:,:,:]) 
        msg_token = msg_token.view(b, self.T, 1, d) 
        
        msg_token = msg_token.permute(1,2,0,3).view(self.T, b, d) 
        # 使用梯度检查点计算 message_attention
        if use_checkpoint:
            attn_out = checkpoint(self.message_attn, self.message_ln(msg_token), self.message_ln(msg_token), self.message_ln(msg_token), need_weights=False)[0]
            msg_token = msg_token + self.drop_path(attn_out)
        else:
            msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token), self.message_ln(msg_token), self.message_ln(msg_token), need_weights=False)[0])
        
        # msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token),self.message_ln(msg_token),self.message_ln(msg_token),need_weights=False)[0])
        msg_token = msg_token.view(self.T, 1, b, d).permute(1,2,0,3)
        
        x = torch.cat([x, msg_token], dim=0)
        
        x = x.view(l+1, -1, d)
        # 使用梯度检查点计算 attention
        if use_checkpoint:
            attn = checkpoint(self.attention, self.ln_1(x))
            x = x + drop_path(attn)
        else:
            x = x + self.drop_path(self.attention(self.ln_1(x)))
        # x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x[:l,:,:]
        if use_checkpoint:
            attn_drop = checkpoint(self.mlp, self.ln_2(x))
            x = x + drop_path(attn_drop)
        else:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        # x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # x: 50 bT c
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, use_checkpoint=False):
        # MHSA
        if use_checkpoint:
            attn_out = checkpoint(self.attention, self.ln_1(x))
        else:
            attn_out = self.attention(self.ln_1(x))
        x = x + self.drop_path(attn_out)
        # FFN
        if use_checkpoint:
            mlp_out = checkpoint(self.mlp, self.ln_2(x))
            x = x + self.drop_path(mlp_out)
        else:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class AIMResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = 0.25
        self.T_Adapter = Adapter(d_model, skip_connect=False)


        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # x: 50 bT c
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, use_checkpoint=False):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        ## temporal adaptation
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=16)
        if use_checkpoint:
            attn_out = checkpoint(self.attention, self.ln_1(xt))
            xt = self.T_Adapter(attn_out)
        else:
            xt = self.T_Adapter(self.attention(self.ln_1(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + self.drop_path(xt)
        ## spatial adaptation
        if use_checkpoint:
            attn_out = checkpoint(self.attention, self.ln_1(x))
            xt = self.S_Adapter(attn_out)
        else:
            xt = self.S_Adapter(self.attention(self.ln_1(xt)))
        x = x + self.drop_path(xt)
        # FFN   joint adaptation
        xn = self.ln_2(x)
        if use_checkpoint:
            mlp_out = checkpoint(self.mlp, xn)
        else:
            mlp_out = self.mlp(xn)
        x = x + self.drop_path(mlp_out) + self.drop_path(self.scale * self.MLP_Adapter(xn))

        return x


class AIMResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = 0.25
        self.T_Adapter = Adapter(d_model, skip_connect=False)


        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # x: 50 bT c
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, use_checkpoint=False):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        ## temporal adaptation
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=16)
        if use_checkpoint:
            attn_out = checkpoint(self.attention, self.ln_1(xt))
            xt = self.T_Adapter(attn_out)
        else:
            xt = self.T_Adapter(self.attention(self.ln_1(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + self.drop_path(xt)
        ## spatial adaptation
        if use_checkpoint:
            attn_out = checkpoint(self.attention, self.ln_1(x))
            xt = self.S_Adapter(attn_out)
        else:
            xt = self.S_Adapter(self.attention(self.ln_1(xt)))
        x = x + self.drop_path(xt)
        # FFN   joint adaptation
        xn = self.ln_2(x)
        if use_checkpoint:
            mlp_out = checkpoint(self.mlp, xn)
        else:
            mlp_out = self.mlp(xn)
        x = x + self.drop_path(mlp_out) + self.drop_path(self.scale * self.MLP_Adapter(xn))

        return x


# 可以进行识别，加入Cross条件的transformer
class V_Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=None, Block = "Origin", T = 8):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)] 
        print('dropout used:{}'.format(dropout))
        self.width = width
        self.layers = layers

        if Block == 'Origin':
            print("model Block: ResidualAttentionBlock")
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]) for i in range(layers)])
        elif Block == 'Cross':
            print("model Block: CrossFramelAttentionBlock")
            # 创建剩余层的ResidualAttentionBlock
            # 假设ResidualAttentionBlock的定义不需要attn_mask和T参数
            residual_blocks = [
                ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]) for i in range(layers-3)
            ]
            # 创建前三层CrossFramelAttentionBlock
            cross_frame_blocks = [
                CrossFramelAttentionBlock(width, heads, attn_mask, dropout[i], T) for i in range(layers-3, layers)
            ]

            # 合并两个列表
            all_blocks =  residual_blocks+ cross_frame_blocks

            # 创建Sequential模型
            self.resblocks = nn.Sequential(*all_blocks)
        self.grad_checkpointing = True

    def forward(self, x: torch.Tensor):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x)
            else:
                x = r(x)
        return x

# 原始的text使用的Transformer
# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=None):
#         super().__init__()
#         if dropout is None:
#             dropout = [0.0 for i in range(layers)] 
#         print('dropout used:{}'.format(dropout))
#         self.width = width
#         self.layers = layers
        
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]) for i in range(layers)])
#         self.grad_checkpointing = True

#     def forward(self, x: torch.Tensor):
#         for r in self.resblocks:
#             if self.grad_checkpointing and not torch.jit.is_scripting():
#                 x = checkpoint(r, x)
#             else:
#                 x = r(x)
#         return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=None):
        """
        整合版 Transformer,支持动态上下文插入 (maple_prompts)、每层独立的 dropout 设置，以及梯度检查点。
        """
        super().__init__()
        self.width = width
        self.layers = layers
        self.attn_mask = attn_mask

        # 初始化 dropout，默认为 0.0
        if dropout is None:
            dropout = [0.0 for _ in range(layers)]
        if len(dropout) != layers:
            raise ValueError("Dropout list length must match the number of layers.")
        print(f'Dropout used for each layer: {dropout}')

        # 用 ModuleList 定义残差块
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]) for i in range(layers)]
        )

        # 梯度检查点开关
        self.grad_checkpointing = True

    def forward(self, x: torch.Tensor, maple_prompts=None):
        """
        参数:
            x: 输入的特征张量。
            maple_prompts: 可选，用于动态插入上下文的提示列表，默认为 None。
        """
        if maple_prompts:
            num_prompts = maple_prompts[0].shape[0]
            for i, blk in enumerate(self.resblocks):
                if i == 0:
                    # 第一层正常处理输入
                    x = blk(x)
                else:
                    # 拆分输入，动态插入上下文
                    prefix = x[:1, :, :]  # 保留首行（一般是 CLS Token）
                    suffix = x[1 + num_prompts:, :, :]  # 跳过插入的 Prompt
                    textual_context = maple_prompts[i - 1]  # 当前层的动态上下文
                    textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2)

                    # 合并上下文并送入残差块
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        # 使用梯度检查点
                        x = checkpoint(blk, x)
                    else:
                        x = blk(x)
        else:
            # 无上下文插入，直接逐层前向计算
            for blk in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # 使用梯度检查点
                    x = checkpoint(blk, x)
                else:
                    x = blk(x)

        return x



class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,dropout = None,joint=False, emb_dropout = 0., Block = "Origin", T=8):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)
        self.emb_dropout = emb_dropout
        self.joint = joint
        if joint:
            print('=====using space-time attention====')
            self.T = T
            self.time_embedding = nn.Parameter(scale * torch.randn(T, width))  # pos emb
        if emb_dropout > 0:
            print('emb_dropout:{}'.format(emb_dropout))

        ## Attention Blocks
        self.transformer = V_Transformer(width, layers, heads, dropout=dropout,Block = Block, T=T,)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        if x.shape[1] == 2:
            conv_2to3 = nn.Conv2d(2, 3, kernel_size=1, bias=True)
            conv_2to3 = conv_2to3.to(x.device)  # 将卷积层移动到输入数据所在设备
            x = conv_2to3(x)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
       
        if self.joint:
            from einops import rearrange
            B = x.shape[0] // self.T
            cls_tokens = x[:B, 0, :].unsqueeze(1)  # only one cls_token
            x = x[:,1:]
            x = rearrange(x, '(b t) n c -> (b n) t c',b=B,t=self.T)
            x = x + self.time_embedding.to(x.dtype)   # temporal pos emb
            x = rearrange(x, '(b n) t c -> b (n t) c',b=B,t=self.T)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 joint=False,
                 tm=None, Block = "Origin", T=8,dropout = 0., emb_dropout = 0.,
                 ):
        super().__init__()
        self.context_length = context_length
        if dropout > 0.:
            dpr = [x.item() for x in torch.linspace(0, dropout, vision_layers)]  # stochastic depth decay rule
        else:
            dpr = None

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )


        else:
            vision_heads = vision_width // 64

            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                joint=joint,dropout=dpr,
                emb_dropout=emb_dropout,
                Block=Block,
                T=T,
            )


        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            dropout=dpr,
        )
        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.beta = nn.Parameter(torch.tensor([0.8, 0.2], dtype=torch.float), requires_grad=True)
        self.dropout = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.T = T

        self.initialize_parameters()


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
                        
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, images, mvs):
        image_feat = self.visual(images.type(self.dtype))
        mvs_feat = self.visual(mvs.type(self.dtype))

        return image_feat, mvs_feat


    def encode_text(self, text, return_token=False):
        # print("encode_text:",text)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # eg, [400 77 512]

        text_token = x @ self.text_projection   # eg, [400 77 512]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection   # 400 512 

        if return_token:
            return x, text_token
        else:
            return x, None    


    def forward(self, image, mv, text, return_token=False):
        image_feats, mv_feats= self.encode_image(image, mv)
        cls_feat, text_feats = self.encode_text(text, return_token)

        return image_feats, mv_feats, cls_feat, text_feats, self.logit_scale.exp()

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict,  tm=None,Block = "Origin", T=8,dropout=0., joint=False,emb_dropout=0.,pretrain=True):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        tm=tm, T=T, joint=joint,
        dropout=dropout, emb_dropout=emb_dropout,Block=Block,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]


    convert_weights(model)
    if pretrain:
        print('loading clip pretrained model!')
        if joint:  #or emb_dropout>0 or dropout>0
            model.load_state_dict(state_dict,strict=False)
        else:
            # 加载状态字典
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # 获取模型的当前 state_dict 中的键
            loaded_keys = [key for key in state_dict.keys() if key not in missing_keys]

            # 输出信息
            # print("Loaded keys:", loaded_keys)
            # print("Missing keys:", missing_keys)
            # print("Unexpected keys:", unexpected_keys)

            # model.load_state_dict(state_dict, strict=False)   # 加载时忽略不匹配的键
            # model.load_state_dict(state_dict)   # 加载时不匹配的键报错
    else:
        print('not using full clip pretrained model, only visual!')
        
        for k in list(state_dict.keys()):
            if not k.find("visual")>-1: 
                state_dict.pop(k)

        model.load_state_dict(state_dict,strict=False)


    # # 输出测试
    # # 获取当前模型的 state_dict
    # model_state_dict = model.state_dict()
    # # 找出匹配成功的键
    # matched_keys = [key for key in state_dict.keys() if key in model_state_dict]

    # # 找出未匹配的键（在state_dict中但在model中不存在）
    # unmatched_keys = [key for key in state_dict.keys() if key not in model_state_dict]

    # # 打印匹配和未匹配的键
    # print("Matched Keys:")   # 512
    # for key in matched_keys:
    #     print(key)

    # print("\nUnmatched Keys (ignored by strict=False):")   # 960
    # for key in unmatched_keys:
    #     print(key)

    # # 可选：打印加载的参数名和形状
    # print("\nLoaded Parameters (Matched Keys):")
    # for key in matched_keys:
    #     print(f"{key}: {model_state_dict[key].shape}")

    # # 输出打印
    return model.eval()



if __name__=='__main__':
    match=ResidualAttentionBlock()
    
    image_input = torch.rand(16,16,768)  # 2 8 3 224 224
    # image_input = image_input.view(2,-1,3)
    text_input = torch.rand(16, 77, 768)
    cls_input = torch.rand(16, 768)
    cls_input = cls_input.unsqueeze(1) 
    cls_input = cls_input + match(cls_input, image_input)
    # input = rearrange(input, 'b t c h w -> b c t h w')

    print(cls_input.shape)
    cls_input = cls_input.squeeze()
    print(cls_input.shape)
    # print(f.shape)
    # print(f)