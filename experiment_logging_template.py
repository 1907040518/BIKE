
import time
import json
import torch
import psutil
import numpy as np
from datetime import datetime
import pandas as pd

class ExperimentLogger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.start_time = None
        self.end_time = None
        self.epoch_data = []
        self.peak_gpu_memory = 0
        
    def start_training(self):
        """开始训练计时"""
        self.start_time = time.time()
        print(f"🚀 Started experiment: {self.experiment_name}")
        
    def log_epoch(self, epoch, train_loss, val_loss, val_acc, learning_rate):
        """记录每个epoch的指标"""
        # 获取GPU内存使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_memory)
        else:
            gpu_memory = 0
            
        epoch_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': learning_rate,
            'gpu_memory_gb': gpu_memory,
            'timestamp': datetime.now().isoformat()
        }
        
        self.epoch_data.append(epoch_info)
        print(f"Epoch {epoch}: Val Acc = {val_acc:.2f}%, GPU Mem = {gpu_memory:.2f}GB")
        
    def end_training(self):
        """结束训练并生成总结"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        if not self.epoch_data:
            print("⚠️ No epoch data recorded!")
            return {}
            
        # 基本统计
        val_accs = [epoch['val_acc'] for epoch in self.epoch_data if epoch['val_acc'] > 0]
        train_losses = [epoch['train_loss'] for epoch in self.epoch_data if epoch['train_loss'] > 0]
        val_losses = [epoch['val_loss'] for epoch in self.epoch_data if epoch['val_loss'] > 0]
        
        best_val_acc = max(val_accs) if val_accs else 0
        final_acc = val_accs[-1] if val_accs else 0
        
        # 找到最佳性能的epoch
        best_epoch = 1
        if val_accs:
            best_idx = val_accs.index(best_val_acc)
            # 找到对应的epoch号
            acc_epochs = [epoch['epoch'] for epoch in self.epoch_data if epoch['val_acc'] > 0]
            if best_idx < len(acc_epochs):
                best_epoch = acc_epochs[best_idx]
        
        # 收敛分析（避免除零错误）
        convergence_epoch = len(self.epoch_data)
        if len(val_accs) >= 5:
            # 寻找收敛点：连续5个epoch准确率变化小于1%
            for i in range(4, len(val_accs)):
                recent_accs = val_accs[i-4:i+1]
                if max(recent_accs) - min(recent_accs) < 1.0:  # 变化小于1%
                    convergence_epoch = i + 1
                    break
        
        # 过拟合趋势分析（修复除零错误）
        overfitting_trend = 0.0
        if len(val_losses) >= 10:
            recent_val_loss = val_losses[-5:]  # 最近5个epoch
            if len(recent_val_loss) >= 2 and recent_val_loss[0] != 0:
                # 只有当第一个值不为0时才计算趋势
                overfitting_trend = (recent_val_loss[-1] - recent_val_loss[0]) / recent_val_loss[0] * 100
        
        # 训练效率
        avg_time_per_epoch = total_time / len(self.epoch_data) if self.epoch_data else 0
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.epoch_data),
            'total_training_time_seconds': total_time,
            'total_training_time_hours': total_time / 3600,
            'avg_time_per_epoch_minutes': avg_time_per_epoch / 60,
            'best_val_acc': best_val_acc,
            'final_acc': final_acc,
            'best_epoch': best_epoch,
            'convergence_epoch': convergence_epoch,
            'peak_gpu_memory_gb': self.peak_gpu_memory,
            'overfitting_trend_percent': overfitting_trend,
            'end_timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ Experiment '{self.experiment_name}' completed!")
        print(f"📊 Best accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"🔥 Peak GPU memory: {self.peak_gpu_memory:.2f} GB")
        
        return summary

def count_parameters(model):
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 详细组件分析
    component_params = {}
    
    for name, param in model.named_parameters():
        # 根据参数名称分类组件
        if 'visual.' in name:
            component = 'visual_encoder'
        elif 'visual_m.' in name:
            component = 'mv_encoder'  
        elif 'visual_r.' in name:
            component = 'residual_encoder'
        elif 'text' in name or 'token_embedding' in name or 'positional_embedding' in name:
            component = 'text_encoder'
        elif 'ln_final' in name or 'text_projection' in name:
            component = 'text_head'
        elif 'logit_scale' in name:
            component = 'logit_scale'
        elif 'Adapter' in name:
            component = 'adapters'
        else:
            component = 'other'
            
        if component not in component_params:
            component_params[component] = {'total': 0, 'trainable': 0}
            
        component_params[component]['total'] += param.numel()
        if param.requires_grad:
            component_params[component]['trainable'] += param.numel()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params,
        'trainable_ratio': (trainable_params / total_params * 100) if total_params > 0 else 0,
        'component_params': component_params
    }

def calculate_model_size(model):
    """计算模型大小（MB）"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def compare_experiments(results_list):
    """对比多个实验结果"""
    if not results_list:
        return pd.DataFrame()
    
    comparison_data = []
    for result in results_list:
        comparison_data.append({
            'Experiment': result.get('experiment_name', 'Unknown'),
            'Best_Accuracy(%)': result.get('best_val_acc', 0),
            'Final_Accuracy(%)': result.get('final_acc', 0),
            'Trainable_Params': result.get('trainable_params', 0),
            'Trainable_Ratio(%)': result.get('trainable_ratio', 0),
            'Training_Time(h)': result.get('total_training_time_hours', 0),
            'Peak_GPU_Memory(GB)': result.get('peak_gpu_memory_gb', 0),
            'Convergence_Epoch': result.get('convergence_epoch', 0),
            'Model_Size(MB)': result.get('model_size_mb', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # 计算效率指标
    if len(df) > 0:
        df['Efficiency_Score'] = df['Best_Accuracy(%)'] / (df['Training_Time(h)'] + 0.1)  # 避免除零
        df['Memory_Efficiency'] = df['Best_Accuracy(%)'] / (df['Peak_GPU_Memory(GB)'] + 0.1)
        df['Parameter_Efficiency'] = df['Best_Accuracy(%)'] / (df['Trainable_Ratio(%)'] + 0.1)
    
    print("\n📈 Experiment Comparison:")
    print(df.to_string(index=False))
    
    return df
