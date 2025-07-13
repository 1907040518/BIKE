import torch

# 加载checkpoint
checkpoint = torch.load("/home/stu_b/BIKE/exps/hmdb51/ViT-B/16/20250711_203840/model_best.pt", map_location='cpu')

print("模型文件中的所有顶级键:")
print("="*50)
for key in checkpoint.keys():
    value = checkpoint[key]
    print(f"{key}: {type(value).__name__}")
    
    # 如果是字典，进一步查看
    if isinstance(value, dict):
        print(f"  包含 {len(value)} 个子项")
        # 显示前几个子项
        for i, (sub_key, sub_value) in enumerate(value.items()):
            if i >= 3:  # 只显示前3个
                print("    ...")
                break
            if isinstance(sub_value, torch.Tensor):
                print(f"    {sub_key}: 张量 {sub_value.shape}")
            else:
                print(f"    {sub_key}: {type(sub_value).__name__}")

print("\n" + "="*50)

# 检查是否有常见的模型权重键名
model_keys = ['model', 'state_dict', 'model_state_dict', 'net', 'network']
for key in model_keys:
    if key in checkpoint:
        print(f"\n找到模型权重键: '{key}'")
        model_weights = checkpoint[key]
        if isinstance(model_weights, dict):
            print(f"包含 {len(model_weights)} 个参数:")
            
            # 查找beta参数
            beta_params = {}
            for name, param in model_weights.items():
                if isinstance(param, torch.Tensor) and 'beta' in name.lower():
                    beta_params[name] = param
            
            if beta_params:
                print(f"\n找到 {len(beta_params)} 个beta参数:")
                for name, param in beta_params.items():
                    print(f"  {name}: {param.shape}")
                    if param.numel() <= 20:  # 如果参数不太多，显示具体值
                        print(f"    值: {param}")
            else:
                print("\n未找到beta参数")
                # 显示所有参数名（查找可能的归一化层参数）
                print("\n所有参数名:")
                for name in list(model_weights.keys())[:20]:  # 显示前20个
                    print(f"  {name}")
                if len(model_weights) > 20:
                    print("  ...")
