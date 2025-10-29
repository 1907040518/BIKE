from collections import OrderedDict
from typing import Tuple, Union
import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


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

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout if dropout is not None else 0.)
        self.ln_1 = LayerNorm(d_model)
        
        # 修复：确保dropout不为None
        dropout_rate = dropout if dropout is not None else 0.
        self.drop_path = DropPath(dropout_rate) if dropout_rate > 0. else nn.Identity()
        
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
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attention(self.ln_1(x)))

        # FFN
        if use_checkpoint:
            mlp_out = checkpoint(self.mlp, self.ln_2(x))
            x = x + self.drop_path(mlp_out)
        else:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        
        return x  # 添加return语句

# 可以进行识别，加入Cross条件的transformer
class V_Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=None, T = 8):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)] 
        print('dropout used:{}'.format(dropout))
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]) for i in range(layers)])

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

class SimpleVideoActionPromptLearner(nn.Module):
    """简化版本:直接从视频特征生成action prompt"""
    def __init__(self, clip_model, hidden_dim=None):
        super().__init__()
        
        ctx_dim = clip_model.transformer.width
        if hidden_dim is None:
            hidden_dim = ctx_dim
        
        # 简单的MLP生成action prompt
        self.prompt_generator = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, ctx_dim)
        )
        
        # 可选:添加一个可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, video_features):
        """
        Args:
            video_features: [B, T, D] 视频特征
        Returns:
            prompt_features: [B, D] 每个视频的prompt特征
        """
        # 时序池化
        video_feat_pooled = video_features.mean(dim=1)  # [B, D]
        
        # 生成prompt
        prompt_features = self.prompt_generator(video_feat_pooled)
        prompt_features = prompt_features * self.scale
        
        return prompt_features
    
class VideoActionPromptLearner(nn.Module):
    """为每个视频学习action prompt,而不是每个类别"""
    def __init__(self, clip_model, n_ctx=4, ctx_init=None):
        super().__init__()
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.transformer.width
        
        # 可学习的context tokens
        if ctx_init:
            # 用特定文本初始化
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            self.n_ctx = n_ctx
        else:
            # 随机初始化
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.n_ctx = n_ctx
        
        self.ctx = nn.Parameter(ctx_vectors)  # 可学习的prompt tokens
        
        # 用于生成视频特定的prompt
        self.meta_net = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(ctx_dim // 2, n_ctx * ctx_dim)
        )
        
        self.clip_model = clip_model
        
    def forward(self, video_features):
        """
        Args:
            video_features: [B, T, D] 视频特征
        Returns:
            prompt_features: [B, D] 每个视频的prompt特征
        """
        B, T, D = video_features.shape
        
        # 使用视频特征生成动态prompt
        video_feat_pooled = video_features.mean(dim=1)  # [B, D]
        
        # 生成视频特定的context偏移
        ctx_shift = self.meta_net(video_feat_pooled)  # [B, n_ctx * D]
        ctx_shift = ctx_shift.view(B, self.n_ctx, D)  # [B, n_ctx, D]
        
        # 基础prompt + 视频特定偏移
        ctx = self.ctx.unsqueeze(0).expand(B, -1, -1)  # [B, n_ctx, D]
        ctx = ctx + ctx_shift  # [B, n_ctx, D]
        
        # 池化得到prompt特征
        prompt_features = ctx.mean(dim=1)  # [B, D]
        
        return prompt_features
    
class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,dropout = None,joint=False, emb_dropout = 0.,T=8):
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
        self.transformer = V_Transformer(width, layers, heads, dropout=dropout,T=T,)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        
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

class ResidualEncoder(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers_to_use, heads, output_dim, dropout=None, emb_dropout=0.):
        """
        独立的残差帧编码器，不共享参数
        Args:
            input_resolution: 输入分辨率
            patch_size: patch大小
            width: 特征维度
            layers_to_use: 要使用的层数列表，例如[0, 1]
            heads: 注意力头数
            output_dim: 输出维度
            dropout: dropout率列表或None
            emb_dropout: embedding dropout率
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.layers_to_use = layers_to_use
        # 处理dropout=None的情况
        if dropout is None:
            # 为每一层创建0.0的dropout
            dropout = [0.0] * len(layers_to_use)
        elif not isinstance(dropout, (list, tuple)):
            # 如果是单个值，转换为列表
            dropout = [dropout] * len(layers_to_use)
        # 独立的卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        # 独立的嵌入层
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)
        self.emb_dropout = emb_dropout
        
        # 独立的transformer层 - 只创建指定数量的层
        num_layers = len(layers_to_use)
        self.transformer_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            # 处理dropout参数
            if dropout is not None and isinstance(dropout, (list, tuple)) and i < len(dropout):
                block_dropout = dropout[i]
            elif dropout is not None and not isinstance(dropout, (list, tuple)):
                block_dropout = dropout
            else:
                block_dropout = 0.
            
            self.transformer_blocks.append(
                ResidualAttentionBlock(width, heads, dropout=block_dropout)
            )
        
        # 独立的最终层
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        # 用于2通道到3通道的转换
        self.conv_2to3 = nn.Conv2d(2, 3, kernel_size=1, bias=True)
    
    def forward(self, x: torch.Tensor):
        # 处理2通道输入
        if x.shape[1] == 2:
            x = self.conv_2to3(x)
            
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], 
                                                                      dtype=x.dtype, device=x.device), x], 
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # 使用独立的transformer层
        for block in self.transformer_blocks:
            if hasattr(block, 'grad_checkpointing') and block.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x)
            else:
                x = block(x)
                
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_post(x[:, 0, :])
        
        if self.proj is not None:
            x = x @ self.proj
        return x

class MVSEncoder(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers_to_use, heads, output_dim, dropout=None, emb_dropout=0.):
        """
        独立的MVS编码器，不共享参数
        Args:
            input_resolution: 输入分辨率
            patch_size: patch大小
            width: 特征维度
            layers_to_use: 要使用的层数列表，例如[0, 1]
            heads: 注意力头数
            output_dim: 输出维度
            dropout: dropout率列表或None
            emb_dropout: embedding dropout率
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.layers_to_use = layers_to_use
        
        # 处理dropout=None的情况
        if dropout is None:
            # 为每一层创建0.0的dropout
            dropout = [0.0] * len(layers_to_use)
        elif not isinstance(dropout, (list, tuple)):
            # 如果是单个值，转换为列表
            dropout = [dropout] * len(layers_to_use)
        
        # 独立的卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        # 独立的嵌入层
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)
        self.emb_dropout = emb_dropout
        
        # 独立的transformer层 - 只创建指定数量的层
        num_layers = len(layers_to_use)
        self.transformer_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            # 处理dropout参数
            if dropout is not None and isinstance(dropout, (list, tuple)) and i < len(dropout):
                block_dropout = dropout[i]
            elif dropout is not None and not isinstance(dropout, (list, tuple)):
                block_dropout = dropout
            else:
                block_dropout = 0.
            
            self.transformer_blocks.append(
                ResidualAttentionBlock(width, heads, dropout=block_dropout)
            )
        
        # 独立的最终层
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        # 用于2通道到3通道的转换 (如果MVS也是2通道的话)
        self.conv_2to3 = nn.Conv2d(2, 3, kernel_size=1, bias=True)
    
    def forward(self, x: torch.Tensor):

        # 处理2通道输入 (如果MVS是2通道的话)
        if x.shape[1] == 2:
            x = self.conv_2to3(x)
            
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], 
                                                                      dtype=x.dtype, device=x.device), x], 
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # 使用独立的transformer层
        for block in self.transformer_blocks:
            if hasattr(block, 'grad_checkpointing') and block.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x)
            else:
                x = block(x)
                
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
                 tm=None, T=8,dropout = 0., emb_dropout = 0.,
                 action_prompt_type='simple',  # 新增: 'simple' 或 'meta'
                 action_prompt_enabled=False,  # 新增: 是否启用
                 residual_layers_to_use=None,  # 新增参数：指定使用哪几层 
                 mvs_layers_to_use=None  # 新增MVS层参数
                ):
        super().__init__()
        self.context_length = context_length
        if dropout > 0.:
            dpr = [x.item() for x in torch.linspace(0, dropout, vision_layers)]  # stochastic depth decay rule
        else:
            dpr = None
        if residual_layers_to_use is None:
            residual_layers_to_use = [0, 1]  # 默认使用前两层

        if mvs_layers_to_use is None:
            mvs_layers_to_use = [0, 1]  # 默认也使用前两层

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            # 为ResNet创建一个轻量级的残差编码器
            self.residual_encoder = None  # 需要单独实现ResNet的轻量版本
            self.mvs_encoder = None


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
                T=T,
            )
            # 创建独立的残差编码器
            residual_dropout = None
            if dpr is not None and residual_layers_to_use is not None:
                # 只取指定层的dropout率，确保索引有效
                residual_dropout = []
                for layer_idx in residual_layers_to_use:
                    if layer_idx < len(dpr):
                        residual_dropout.append(dpr[layer_idx])
                    else:
                        residual_dropout.append(0.)  # 默认值
            
            self.residual_encoder = ResidualEncoder(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers_to_use=residual_layers_to_use,
                heads=vision_heads,
                output_dim=embed_dim,
                dropout=residual_dropout,
                emb_dropout=emb_dropout
            )
        
            # 创建独立的MVS编码器
            mvs_dropout = None
            if dpr is not None and mvs_layers_to_use is not None:
                mvs_dropout = []
                for layer_idx in mvs_layers_to_use:
                    if layer_idx < len(dpr):
                        mvs_dropout.append(dpr[layer_idx])
                    else:
                        mvs_dropout.append(0.)
            
            self.mvs_encoder = MVSEncoder(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers_to_use=mvs_layers_to_use,
                heads=vision_heads,
                output_dim=embed_dim,
                dropout=mvs_dropout,
                emb_dropout=emb_dropout
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
        self.beta = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float), requires_grad=True)

        self.dropout = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.T = T
        # 初始化action prompt learner
        self.action_prompt_learner = None
        self.action_prompt_enabled = action_prompt_enabled
        
        if action_prompt_enabled:
            if action_prompt_type == 'simple':
                self.action_prompt_learner = SimpleVideoActionPromptLearner(self)
            elif action_prompt_type == 'meta':
                self.action_prompt_learner = VideoActionPromptLearner(self, n_ctx=4)
            else:
                raise ValueError(f"Unknown action_prompt_type: {action_prompt_type}")
        
        self.initialize_parameters()
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
        
        # 初始化action prompt learner
        if self.action_prompt_learner is not None:
            for module in self.action_prompt_learner.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

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


    def encode_image(self, images, res,mv):
        # 编码原始图像
        image_feat = self.visual(images.type(self.dtype))
        
        # 编码残差信息
        if self.residual_encoder is not None:
            res_feat = self.residual_encoder(res.type(self.dtype))
            mvs_feats = self.mvs_encoder(res.type(self.dtype))
        else:
            # 如果没有专门的残差编码器(例如在ResNet的情况下)，则回退到使用完整的编码器
            res_feat = self.visual(res.type(self.dtype))
        
        
        return image_feat, res_feat, mvs_feats


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


    def forward(self, image, residual,mv,text, return_token=False):
        image_feats, residual_feats, mvs_feats= self.encode_image(image, residual, mv)
        cls_feat, text_feats = self.encode_text(text, return_token)
        # 使用可学习的beta参数，并确保它参与反向传播
        weights = F.softmax(self.beta, dim=0)  # 计算权重，确保数值范围正常

        merged_feats = weights[0] * image_feats + weights[1] * residual_feats + weights[2] * mvs_feats

        # 🔧 新增: 从视频特征生成action prompt
        if self.action_prompt_learner is not None:
            action_prompt = self.action_prompt_learner(merged_feats)  # [B, D]
            # 将action prompt加到视频特征上
            merged_feats = merged_feats + action_prompt.unsqueeze(1)  # broadcast到所有帧

        # Reshape回 [B*T, D]
        merged_feats = merged_feats.view(-1, merged_feats.shape[-1])

        return merged_feats, cls_feat, text_feats, self.logit_scale.exp()

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


def build_model(state_dict: dict,  tm=None, T=8,dropout=0., joint=False,emb_dropout=0.,pretrain=True,residual_layers_to_use=None, mvs_layers_to_use=None, action_prompt_type='simple', action_prompt_enabled=False):
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
        dropout=dropout, emb_dropout=emb_dropout,residual_layers_to_use=residual_layers_to_use,mvs_layers_to_use=mvs_layers_to_use, action_prompt_type=action_prompt_type, action_prompt_enabled=action_prompt_enabled
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

    else:
        print('not using full clip pretrained model, only visual!')
        
        for k in list(state_dict.keys()):
            if not k.find("visual")>-1: 
                state_dict.pop(k)

        model.load_state_dict(state_dict,strict=False)


 
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