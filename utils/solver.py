import torch.optim as optim
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR

def _optimizer(config, model, video_head, extra_params=None):

    params = list(model.parameters()) + list(video_head.parameters())
    if extra_params is not None:
        params.extend(list(extra_params))

    if config.solver.optim == 'adam':
        optimizer = optim.Adam([{'params': model.parameters()},  
         {'params': video_head.parameters(), 'lr': config.solver.lr}],
                               lr=config.solver.lr * config.solver.clip_ratio, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=0.2)
        print('Adam')
    elif config.solver.optim == 'sgd':
        optimizer = optim.SGD([{'params': model.parameters()},  
         {'params': video_head.parameters(), 'lr': config.solver.lr}],
                              config.solver.lr * config.solver.clip_ratio,
                              momentum=config.solver.momentum,
                              weight_decay=config.solver.weight_decay)
        print('SGD')
    elif config.solver.optim == 'adamw':
        vision_params = []
        text_params = []
        fusion_params = [] # 存放你的创新模块

        # 计算不同部分的基础学习率
        backbone_lr = config.solver.lr * config.solver.clip_ratio
        # 优先从 config.network.attribute_guided_fusion 读取 fusion_lr_scale，
        # 若不存在则回退到 config.network.textcomp，再回退到默认值 10.0
        fusion_scale = None
        try:
            # 支持 dict-like 或 DotMap
            agf = None
            if isinstance(config.network, dict):
                agf = config.network.get('attribute_guided_fusion')
            else:
                agf = config.network.get('attribute_guided_fusion', None)

            if agf is not None:
                if hasattr(agf, 'get'):
                    fusion_scale = agf.get('fusion_lr_scale', None)
                else:
                    fusion_scale = getattr(agf, 'fusion_lr_scale', None)
        except Exception:
            fusion_scale = None

        if fusion_scale is None:
            try:
                fusion_scale = config.network.get('textcomp', {}).get('fusion_lr_scale', 10.0)
            except Exception:
                fusion_scale = 10.0

        try:
            fusion_lr = backbone_lr * float(fusion_scale)
        except Exception:
            fusion_lr = backbone_lr * 10.0

        # 打印所用的 fusion_lr_scale 来源，便于排查 YAML 与代码一致性
        try:
            src = 'attribute_guided_fusion' if (isinstance(config.network, dict) and config.network.get('attribute_guided_fusion') and config.network.get('attribute_guided_fusion').get('fusion_lr_scale') is not None) or (not isinstance(config.network, dict) and getattr(config.network, 'attribute_guided_fusion', None) and ((hasattr(config.network.attribute_guided_fusion, 'get') and config.network.attribute_guided_fusion.get('fusion_lr_scale', None) is not None) or getattr(config.network.attribute_guided_fusion, 'fusion_lr_scale', None) is not None)) else ('textcomp' if (isinstance(config.network, dict) and config.network.get('textcomp') and config.network.get('textcomp').get('fusion_lr_scale') is not None) or (not isinstance(config.network, dict) and getattr(config.network, 'textcomp', None) and ((hasattr(config.network.textcomp, 'get') and config.network.textcomp.get('fusion_lr_scale', None) is not None) or getattr(config.network.textcomp, 'fusion_lr_scale', None) is not None)) else 'default')
        except Exception:
            src = 'default'
        print(f'Using fusion_lr_scale={fusion_scale} (source={src})')

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue # 跳过依然被冻结的参数
                
            # 🔥 1. 拦截你的动态融合模块 (使用高学习率)
            if 'attribute_guided_fusion' in name:
                fusion_params.append(param)
                
            # 🔥 2. 拦截视觉主干以及你自定义的两个视频编码器 (使用保护性的低学习率)
            elif any(k in name for k in ['visual', 'residual_encoder', 'mvs_encoder']):
                vision_params.append(param)
                
            # 🔥 3. 剩下的归为文本分支 (包含 text_projection, positional_embedding等)
            else:
                text_params.append(param)

        # 构建分层学习率的优化器组
        optimizer_groups = [
            {'params': vision_params, 'lr': backbone_lr},
            {'params': text_params, 'lr': backbone_lr},
            {'params': fusion_params, 'lr': fusion_lr},
            {'params': video_head.parameters(), 'lr': fusion_lr} # 预测头同样需要高学习率
        ]
        
        if extra_params is not None:
            optimizer_groups.append({'params': list(extra_params), 'lr': fusion_lr})

        optimizer = optim.AdamW(
            optimizer_groups,
            lr=backbone_lr,
            betas=(0.9, 0.999), 
            eps=1e-8,
            weight_decay=config.solver.weight_decay
        )
        print(f'AdamW Initialized: Backbone LR={backbone_lr:.2e}, Fusion/Head LR={fusion_lr:.2e}')
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.solver.optim))
    return optimizer



def _lr_scheduler(config, optimizer):
    if config.solver.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.solver.epochs,
            warmup_epochs=config.solver.lr_warmup_step
        )
    elif config.solver.type == 'multistep':
        if isinstance(config.solver.lr_decay_step, list):
            milestones = config.solver.lr_decay_step
        elif isinstance(config.solver.lr_decay_step, int):
            milestones = [
                config.solver.lr_decay_step * (i + 1)
                for i in range(config.solver.epochs //
                               config.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config.solver.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config.solver.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.solver.type))
    return lr_scheduler


