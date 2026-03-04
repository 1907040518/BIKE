"""
TextComp: Instance-Aware Dynamic Video-Text Interaction Module

Core innovation:
1. Semantic Prior Bank: LLM-generated attributes as learnable priors
2. Instance-Aware Cross-Modal Attention: each visual modality independently queries semantic priors
3. Dynamic Router: adaptive residual fusion weights based on multi-modal context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    """
    单个视觉模态对语义先验的交叉注意力。
    Visual features (Query) attend to Semantic Priors (Key/Value).
    
    输入:
        visual_feat: (B, D)  — 某一模态的视觉特征 (CLS token)
        semantic_prior: (N_attr, D) 或 (B, N_attr, D) — 语义属性嵌入
    输出:
        attended_feat: (B, D) — 语义增强后的视觉特征
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5
        
        # Query from visual, Key/Value from semantic
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Gated residual: 学习"应该融合多少语义信息"
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, visual_feat, semantic_prior):
        """
        Args:
            visual_feat: (B, D) 
            semantic_prior: (N, D) or (B, N, D)
        Returns:
            out: (B, D) 语义增强的视觉特征
        """
        B, D = visual_feat.shape
        
        # 处理 semantic_prior 维度
        if semantic_prior.dim() == 2:
            # (N, D) -> (B, N, D) broadcast
            semantic_prior = semantic_prior.unsqueeze(0).expand(B, -1, -1)
        
        N = semantic_prior.shape[1]
        
        # Normalize
        q = self.norm_q(visual_feat).unsqueeze(1)  # (B, 1, D)
        kv = self.norm_kv(semantic_prior)           # (B, N, D)
        
        # Project
        q = self.q_proj(q).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, 1, d)
        k = self.k_proj(kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        v = self.v_proj(kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, 1, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, D)  # (B, D)
        out = self.out_proj(out)
        
        # Gated residual: 自适应决定融入多少语义信息
        gate_input = torch.cat([visual_feat, out], dim=-1)  # (B, 2D)
        gate_weight = self.gate(gate_input)                  # (B, D), values in [0, 1]
        
        result = visual_feat + gate_weight * out  # 残差连接 + 门控
        
        return result


class SemanticPriorBank(nn.Module):
    """
    语义先验库：将 CLIP 文本编码器产生的属性嵌入转化为可学习的先验。
    
    这个模块接收 text encoder 的输出，并通过一个轻量变换使其成为
    可查询的 key-value 对。
    
    关键设计：
    - 不直接使用 frozen text features，而是加一个可学习的 adaptation layer
    - 这样语义先验可以在训练中逐步适应视觉特征空间
    """
    def __init__(self, embed_dim, num_attributes=16, dropout=0.1):
        super().__init__()
        self.num_attributes = num_attributes
        
        # 轻量适配层：让语义先验适应视觉空间
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        # 残差缩放因子，初始化为较小值，保证训练初期先验接近原始文本嵌入
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, text_features):
        """
        Args:
            text_features: (N_class, D) 或 (B, N_class, D) 文本特征
        Returns:
            semantic_priors: 同形状的适配后语义先验
        """
        adapted = self.adapter(text_features)
        # 残差连接 + 可学习缩放
        priors = text_features + self.residual_scale * adapted
        priors = self.norm(priors)
        return priors


class DynamicRouter(nn.Module):
    """
    动态路由器：根据融合后的多模态上下文，为每个样本计算自适应的模态融合权重。
    
    关键设计：
    1. 不是简单的 softmax 权重，而是生成 "残差权重"
       - base_weight = [1/3, 1/3, 1/3] (均匀)
       - dynamic_residual = Router(context)  (实例自适应调整)
       - final_weight = base_weight + residual_scale * dynamic_residual
    2. 这保证了训练初期模型行为接近均匀融合，稳定训练
    3. 同时引入温度参数控制权重的尖锐度
    """
    def __init__(self, embed_dim, num_modalities=3, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // 2
        
        self.num_modalities = num_modalities
        
        # 上下文编码器：融合三个模态的语义增强特征
        self.context_encoder = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 权重预测头
        self.weight_head = nn.Linear(hidden_dim, num_modalities)
        
        # 可学习温度参数（控制权���分布的尖锐度）
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 残差缩放：初始化为小值，训练初期接近均匀权重
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        # 基础权重（均匀分布）
        self.register_buffer('base_weight', torch.ones(num_modalities) / num_modalities)
    
    def forward(self, feat_iframe, feat_residual, feat_mv):
        """
        Args:
            feat_iframe:   (B, D) 语义增强后的 I-frame 特征
            feat_residual: (B, D) 语义增强后的 Residual 特征
            feat_mv:       (B, D) 语义增强后的 MV 特征
        Returns:
            weights: (B, 3) 每个样本的模态融合权重
            context: (B, hidden_dim) 上下文特征（可选用于其他目的）
        """
        # 拼接三个模态特征作为上下文
        context_input = torch.cat([feat_iframe, feat_residual, feat_mv], dim=-1)  # (B, 3D)
        context = self.context_encoder(context_input)  # (B, hidden_dim)
        
        # 预测动态残差权重
        weight_logits = self.weight_head(context)  # (B, 3)
        
        # 温度控制的 softmax 得到残差调整
        temperature = self.temperature.clamp(min=0.1)  # 防止温度过小
        dynamic_residual = F.softmax(weight_logits / temperature, dim=-1)  # (B, 3)
        
        # 最终权重 = 基础权重 + 可学习缩放 * (动态残差 - 基础权重)
        # 这样初始时 residual_scale ≈ 0，权重接近均匀
        weights = self.base_weight.unsqueeze(0) + self.residual_scale * (dynamic_residual - self.base_weight.unsqueeze(0))
        
        # 归一化确保权重和为1
        weights = F.softmax(weights, dim=-1)
        
        return weights, context


class InstanceAwareDynamicFusion(nn.Module):
    """
    TextComp 的核心模块：Instance-Aware Dynamic Fusion
    
    Pipeline:
    1. Semantic Prior Bank: 适配文本语义先验
    2. Cross-Modal Attention x3: 每个视觉模态独立查询语义先验
    3. Dynamic Router: 根据语义增强的多模态上下文计算��适应权重
    4. Weighted Fusion: 加权融合得到最终视频表示
    
    设计原则：
    - 初期行为接近简单加权平均（beta），保证稳定训练
    - 随训练进行，逐步学习实例自适应的融合策略
    - 语义先验提供"该关注什么"的指导
    """
    def __init__(self, embed_dim, num_heads=8, num_attributes=16, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. 语义先验库
        self.semantic_prior_bank = SemanticPriorBank(
            embed_dim=embed_dim,
            num_attributes=num_attributes,
            dropout=dropout
        )
        
        # 2. 三个独立的交叉模态注意力（每个模态一个）
        self.iframe_cross_attn = CrossModalAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.residual_cross_attn = CrossModalAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.mv_cross_attn = CrossModalAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        
        # 3. 动态路由器
        self.dynamic_router = DynamicRouter(
            embed_dim=embed_dim, 
            num_modalities=3,
            dropout=dropout
        )
        
        # 4. 最终投影（可选，将融合特征映射回原始空间）
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # 全局残差缩放：控制整个模块的影响力
        # 初始化为小值，训练初期几乎不改变原有行为
        self.module_scale = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        """谨慎初始化，保证训练初期稳定"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, iframe_feat, residual_feat, mv_feat, text_cls_feat, 
                simple_fused=None):
        """
        Args:
            iframe_feat:   (B, T, D) I-frame 视觉特征 (来自 VisualTransformer)
            residual_feat: (B, T, D) Residual 视觉特征 (来自 ResidualEncoder)
            mv_feat:       (B, T, D) MV 视觉特征 (来自 MVSEncoder)
            text_cls_feat: (N_class, D) 文本分类特征 (来自 text encoder)
            simple_fused:  (B, T, D) 简单加权融合的结果 (beta fusion)，用于残差连接
            
        Returns:
            fused_feat:    (B, T, D) 实例自适应融合后的特征
            aux_info:      dict, 包含路由权重等辅助信息（用于可视化/分析）
        """
        B, T, D = iframe_feat.shape
        
        # ============================================
        # Step 1: 构建语义先验
        # ============================================
        # text_cls_feat: (N_class, D) -> 适配后的语义先验
        semantic_priors = self.semantic_prior_bank(text_cls_feat)  # (N_class, D)
        
        # ============================================
        # Step 2: 帧级别的交叉模态注意力
        # 将 (B, T, D) reshape 为 (B*T, D) 进行逐帧处理
        # ============================================
        iframe_flat = iframe_feat.reshape(B * T, D)       # (B*T, D)
        residual_flat = residual_feat.reshape(B * T, D)   # (B*T, D)
        mv_flat = mv_feat.reshape(B * T, D)               # (B*T, D)
        
        # 每个模态独立查询语义先验
        iframe_enhanced = self.iframe_cross_attn(iframe_flat, semantic_priors)      # (B*T, D)
        residual_enhanced = self.residual_cross_attn(residual_flat, semantic_priors) # (B*T, D)
        mv_enhanced = self.mv_cross_attn(mv_flat, semantic_priors)                  # (B*T, D)
        
        # ============================================
        # Step 3: 动态路由
        # 使用时间平均的特征计算样本级别的权重
        # ============================================
        # Reshape back: (B*T, D) -> (B, T, D)
        iframe_enhanced_bt = iframe_enhanced.reshape(B, T, D)
        residual_enhanced_bt = residual_enhanced.reshape(B, T, D)
        mv_enhanced_bt = mv_enhanced.reshape(B, T, D)
        
        # 时间平均，得到样本级表示
        iframe_pooled = iframe_enhanced_bt.mean(dim=1)     # (B, D)
        residual_pooled = residual_enhanced_bt.mean(dim=1)  # (B, D)
        mv_pooled = mv_enhanced_bt.mean(dim=1)              # (B, D)
        
        # 动态路由计算权重
        weights, router_context = self.dynamic_router(
            iframe_pooled, residual_pooled, mv_pooled
        )  # weights: (B, 3)
        
        # ============================================
        # Step 4: 加权融合
        # ============================================
        # 扩展权重到帧级别: (B, 3) -> (B, 1, 1) for broadcasting
        w_iframe = weights[:, 0].unsqueeze(-1).unsqueeze(-1)     # (B, 1, 1)
        w_residual = weights[:, 1].unsqueeze(-1).unsqueeze(-1)   # (B, 1, 1)
        w_mv = weights[:, 2].unsqueeze(-1).unsqueeze(-1)         # (B, 1, 1)
        
        # 加权求和
        dynamic_fused = (w_iframe * iframe_enhanced_bt + 
                        w_residual * residual_enhanced_bt + 
                        w_mv * mv_enhanced_bt)  # (B, T, D)
        
        # 最终投影
        dynamic_fused = self.output_proj(dynamic_fused)  # (B, T, D)
        
        # ============================================
        # Step 5: 全局残差连接
        # 如果提供了简单融合结果，用残差方式叠加
        # 这保证了训练初期行为接近原始 beta 加权
        # ============================================
        if simple_fused is not None:
            fused_feat = simple_fused + self.module_scale * dynamic_fused
        else:
            fused_feat = dynamic_fused
        
        # 收集辅助信息
        aux_info = {
            'routing_weights': weights.detach(),           # (B, 3)
            'module_scale': self.module_scale.detach(),    # scalar
            'temperature': self.dynamic_router.temperature.detach(),
        }
        
        return fused_feat, aux_info