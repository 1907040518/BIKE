import torch
import torch.nn as nn

def analyze_model_parameters(model, model_name="Model"):
    """
    详细分析模型参数
    """
    print(f"🔍 分析模型: {model_name}")
    print("=" * 60)
    
    # 基本统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"📊 参数统计:")
    print(f"   🔢 总参数量: {total_params:,}")
    print(f"   🎯 可训练参数: {trainable_params:,}")
    print(f"   🧊 冻结参数: {frozen_params:,}")
    print(f"   📈 可训练比例: {(trainable_params/total_params*100):.2f}%")
    print()
    
    # 按模块分析
    print("📋 按模块分析:")
    module_stats = {}
    
    for name, param in model.named_parameters():
        # 提取模块名称
        module_name = name.split('.')[0] if '.' in name else name
        
        if module_name not in module_stats:
            module_stats[module_name] = {
                'total': 0,
                'trainable': 0,
                'frozen': 0
            }
        
        param_count = param.numel()
        module_stats[module_name]['total'] += param_count
        
        if param.requires_grad:
            module_stats[module_name]['trainable'] += param_count
        else:
            module_stats[module_name]['frozen'] += param_count
    
    # 按参数量排序
    sorted_modules = sorted(module_stats.items(), 
                          key=lambda x: x[1]['total'], reverse=True)
    
    for module_name, stats in sorted_modules:
        total = stats['total']
        trainable = stats['trainable']
        frozen = stats['frozen']
        trainable_ratio = (trainable / total * 100) if total > 0 else 0
        
        print(f"   📦 {module_name}:")
        print(f"      • 总参数: {total:,}")
        print(f"      • 可训练: {trainable:,} ({trainable_ratio:.1f}%)")
        print(f"      • 冻结: {frozen:,}")
        print()
    
    # 参数类型分析
    print("🎭 参数类型分析:")
    param_types = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name.lower():
            param_type = 'weights'
        elif 'bias' in name.lower():
            param_type = 'biases'
        elif 'norm' in name.lower() or 'bn' in name.lower():
            param_type = 'normalization'
        elif 'embed' in name.lower():
            param_type = 'embeddings'
        else:
            param_type = 'others'
        
        if param_type not in param_types:
            param_types[param_type] = {'total': 0, 'trainable': 0}
        
        param_count = param.numel()
        param_types[param_type]['total'] += param_count
        
        if param.requires_grad:
            param_types[param_type]['trainable'] += param_count
    
    for param_type, stats in param_types.items():
        total = stats['total']
        trainable = stats['trainable']
        trainable_ratio = (trainable / total * 100) if total > 0 else 0
        
        print(f"   🏷️  {param_type.capitalize()}:")
        print(f"      • 总数: {total:,}")
        print(f"      • 可训练: {trainable:,} ({trainable_ratio:.1f}%)")
    
    print()
    
    # 计算模型大小
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    trainable_size_mb = sum(p.nelement() * p.element_size() 
                           for p in model.parameters() if p.requires_grad) / (1024 ** 2)
    
    print(f"💾 模型大小:")
    print(f"   📏 总大小: {total_size_mb:.2f} MB")
    print(f"   🎯 可训练部分: {trainable_size_mb:.2f} MB")
    print(f"   💿 节省空间: {total_size_mb - trainable_size_mb:.2f} MB")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_ratio': trainable_params/total_params*100,
        'total_size_mb': total_size_mb,
        'trainable_size_mb': trainable_size_mb,
        'module_stats': module_stats,
        'param_types': param_types
    }

def quick_param_check(model):
    """
    快速检查参数统计
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"⚡ 快速参数检查:")
    print(f"   🔢 总参数: {total:,}")
    print(f"   🎯 可训练: {trainable:,}")
    print(f"   📊 比例: {trainable/total*100:.2f}%")
    
    return {'total': total, 'trainable': trainable, 'ratio': trainable/total*100}

# 使用示例
if __name__ == "__main__":
    # 创建一个示例模型进行测试
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(1000, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
            self.classifier = nn.Linear(256, 10)
            
            # 冻结backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    model = TestModel()
    analyze_model_parameters(model, "Test PEFT Model")