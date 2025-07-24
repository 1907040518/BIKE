import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from utils.utils import AverageMeter, reduce_tensor, accuracy, gather_labels
class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )
def validate_with_timing(epoch, val_loader, classes, device, model, video_head, mv_head, config, n_class, logger, return_sim=False):
    """
    🚀 带性能测试的验证函数
    测量数据加载时间和推理时间
    
    Args:
        epoch: 当前epoch
        val_loader: 验证数据加载器
        classes: 类别文本特征
        device: 设备
        model: 主模型
        video_head: 视频头
        mv_head: 运动矢量头
        config: 配置
        n_class: 类别数
        logger: 日志记录器
        return_sim: 是否返回相似度
    
    Returns:
        top1.avg: Top-1准确率
        sims_list: 相似度列表 (如果return_sim=True)
        labels_list: 标签列表 (如果return_sim=True)
        timing_stats: 性能统计信息
    """
    top1 = AverageMeter()
    top5 = AverageMeter()
    sims_list = []
    labels_list = []
    
    # 🕒 性能计时器
    data_load_times = AverageMeter()      # 数据加载时间
    inference_times = AverageMeter()      # 推理时间
    feature_extract_times = AverageMeter() # 特征提取时间
    similarity_times = AverageMeter()     # 相似度计算时间
    
    model.eval()
    video_head.eval()
    mv_head.eval()
    
    # 记录总体开始时间
    total_start_time = time.time()
    
    with torch.no_grad():
        # 📝 文本特征预计算
        text_start_time = time.time()
        text_inputs = classes.to(device)  # [n_cls, 77]
        cls_feature, text_features = model.module.encode_text(text_inputs, return_token=True)  # [n_cls, feat_dim]
        text_encode_time = time.time() - text_start_time
        logger.info(f"🔤 Text encoding time: {text_encode_time:.4f}s")
        
        # 🔄 数据加载和推理循环
        data_iter_start = time.time()
        
        for i, (image, mv, residual, class_id) in enumerate(val_loader):
            # ⏱️ 数据加载时间测量
            data_load_end = time.time()
            batch_data_load_time = data_load_end - data_iter_start
            data_load_times.update(batch_data_load_time)
            
            # 🚀 推理开始
            inference_start = time.time()
            
            # 📊 数据预处理
            preprocess_start = time.time()
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])  # b t 3 h w
            mv = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])  # b t 2 h w
            residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])  # b t 3 h w
            
            b, t, c_i, h, w = image.size()
            b, t, c_m, h, w = mv.size()
            
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c_i, h, w)
            mv_input = mv.to(device).view(-1, c_m, h, w)
            residual_input = residual.to(device).view(-1, c_i, h, w)
            preprocess_time = time.time() - preprocess_start
            
            # 🧠 特征提取
            feature_start = time.time()
            image_features, mv_features, res_features = model.module.encode_image(
                image_input, mv_input, residual_input)
            
            # 特征融合
            weights = F.softmax(model.module.beta, dim=0)
            merged_feats = weights[0] * image_features + weights[1] * res_features
            
            mv_features = mv_features.view(b, t, -1)
            merged_feats = merged_feats.view(b, t, -1)
            feature_extract_end = time.time()
            feature_extract_times.update(feature_extract_end - feature_start)
            
            # 🎯 相似度计算
            similarity_start = time.time()
            similarity = video_head(merged_feats, text_features, cls_feature)
            similarity_mv = mv_head(mv_features, text_features, cls_feature)
            
            # 双头融合
            combined_similarity = 0.4 * similarity + 0.6 * similarity_mv
            final_similarity = combined_similarity
            final_similarity = final_similarity.view(b, -1, n_class).softmax(dim=-1)  # [bs, n_frames, n_cls]
            final_similarity = final_similarity.mean(dim=1, keepdim=False)  # [bs, n_cls]
            similarity_end = time.time()
            similarity_times.update(similarity_end - similarity_start)
            
            # 🚀 推理结束
            inference_end = time.time()
            batch_inference_time = inference_end - inference_start
            inference_times.update(batch_inference_time)
            
            # 📊 相似度收集
            if return_sim:
                sims = AllGather(final_similarity)
                labels = gather_labels(class_id)
                sims_list.append(sims)
                labels_list.append(labels)
            
            # 📈 准确率计算
            prec = accuracy(final_similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])
            
            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))
            
            # 📝 定期日志输出
            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                     'DataLoad {data_time.val:.4f}s ({data_time.avg:.4f}s)\t'
                     'Inference {inf_time.val:.4f}s ({inf_time.avg:.4f}s)\t'
                     'FPS {fps:.2f}'.format(
                         i, len(val_loader), top1=top1, top5=top5,
                         data_time=data_load_times, inf_time=inference_times,
                         fps=1.0/inference_times.avg if inference_times.avg > 0 else 0)))
            
            # ⏱️ 准备下一次数据加载时间测量
            data_iter_start = time.time()
    
    # 📊 总体性能统计
    total_time = time.time() - total_start_time
    
    # 🎯 性能统计信息
    timing_stats = {
        'total_time': total_time,
        'avg_data_load_time': data_load_times.avg,
        'avg_inference_time': inference_times.avg,
        'avg_feature_extract_time': feature_extract_times.avg,
        'avg_similarity_time': similarity_times.avg,
        'fps': 1.0 / inference_times.avg if inference_times.avg > 0 else 0,
        'total_samples': len(val_loader) * config.data.batch_size,
        'throughput': (len(val_loader) * config.data.batch_size) / total_time
    }
    
    # 📈 详细性能报告
    logger.info('\n' + '='*80)
    logger.info('🚀 PERFORMANCE ANALYSIS REPORT')
    logger.info('='*80)
    logger.info(f'📊 Accuracy Results:')
    logger.info(f'   • Top-1 Accuracy: {top1.avg:.3f}%')
    logger.info(f'   • Top-5 Accuracy: {top5.avg:.3f}%')
    logger.info(f'\n⏱️  Timing Analysis:')
    logger.info(f'   • Total Test Time: {total_time:.2f}s')
    logger.info(f'   • Average Data Loading Time: {data_load_times.avg*1000:.2f}ms per batch')
    logger.info(f'   • Average Inference Time: {inference_times.avg*1000:.2f}ms per batch')
    logger.info(f'   • Average Feature Extraction Time: {feature_extract_times.avg*1000:.2f}ms per batch')
    logger.info(f'   • Average Similarity Computation Time: {similarity_times.avg*1000:.2f}ms per batch')
    logger.info(f'\n🚀 Performance Metrics:')
    logger.info(f'   • Inference FPS: {timing_stats["fps"]:.2f} batches/second')
    logger.info(f'   • Throughput: {timing_stats["throughput"]:.2f} samples/second')
    logger.info(f'   • Total Samples Processed: {timing_stats["total_samples"]}')
    logger.info(f'\n💡 Per Video Analysis (assuming batch_size={config.data.batch_size}):')
    logger.info(f'   • Data Loading Time per Video: {data_load_times.avg/config.data.batch_size*1000:.2f}ms')
    logger.info(f'   • Inference Time per Video: {inference_times.avg/config.data.batch_size*1000:.2f}ms')
    logger.info(f'   • Total Processing Time per Video: {(data_load_times.avg + inference_times.avg)/config.data.batch_size*1000:.2f}ms')
    logger.info('='*80)
    
    # 📊 原始验证结果日志
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5)))
    
    if return_sim:
        return top1.avg, sims_list, labels_list, timing_stats
    else:
        return top1.avg, None, None, timing_stats


def benchmark_data_loading(val_loader, config, logger, num_batches=10):
    """
    🔍 专门测试数据加载性能的函数
    
    Args:
        val_loader: 验证数据加载器
        config: 配置
        logger: 日志记录器
        num_batches: 测试的批次数量
    
    Returns:
        loading_stats: 数据加载统计信息
    """
    logger.info(f'\n🔍 BENCHMARKING DATA LOADING PERFORMANCE')
    logger.info(f'Testing {num_batches} batches...')
    
    load_times = []
    preprocess_times = []
    
    data_iter_start = time.time()
    
    for i, (image, mv, residual, class_id) in enumerate(val_loader):
        if i >= num_batches:
            break
            
        # 📊 数据加载时间
        load_end = time.time()
        load_time = load_end - data_iter_start
        load_times.append(load_time)
        
        # 📊 预处理时间
        preprocess_start = time.time()
        
        # 模拟预处理操作
        image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
        mv = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])
        residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])
        
        b, t, c_i, h, w = image.size()
        
        preprocess_end = time.time()
        preprocess_time = preprocess_end - preprocess_start
        preprocess_times.append(preprocess_time)
        
        logger.info(f'Batch {i+1}/{num_batches}: Load={load_time*1000:.2f}ms, Preprocess={preprocess_time*1000:.2f}ms')
        
        # 准备下一次测量
        data_iter_start = time.time()
    
    # 📊 统计结果
    avg_load_time = sum(load_times) / len(load_times)
    avg_preprocess_time = sum(preprocess_times) / len(preprocess_times)
    
    loading_stats = {
        'avg_load_time': avg_load_time,
        'avg_preprocess_time': avg_preprocess_time,
        'total_avg_time': avg_load_time + avg_preprocess_time,
        'load_times': load_times,
        'preprocess_times': preprocess_times
    }
    
    logger.info(f'\n📊 DATA LOADING BENCHMARK RESULTS:')
    logger.info(f'   • Average Loading Time: {avg_load_time*1000:.2f}ms per batch')
    logger.info(f'   • Average Preprocessing Time: {avg_preprocess_time*1000:.2f}ms per batch')
    logger.info(f'   • Total Average Time: {(avg_load_time + avg_preprocess_time)*1000:.2f}ms per batch')
    logger.info(f'   • Per Video Loading Time: {avg_load_time/config.data.batch_size*1000:.2f}ms')
    logger.info(f'   • Per Video Total Time: {(avg_load_time + avg_preprocess_time)/config.data.batch_size*1000:.2f}ms')
    
    return loading_stats


# 🎯 使用示例
def example_usage():
    """
    📝 使用示例
    """
    # 在训练/测试脚本中使用
    
    # 1. 使用带性能测试的验证函数
    top1, sims, labels, timing_stats = validate_with_timing(
        epoch, val_loader, classes, device, 
        model, video_head, mv_head, config, n_class, logger, 
        return_sim=False
    )
    
    # 2. 单独测试数据加载性能
    loading_stats = benchmark_data_loading(val_loader, config, logger, num_batches=20)
    
    # 3. 保存性能统计到文件
    import json
    performance_report = {
        'timing_stats': timing_stats,
        'loading_stats': loading_stats,
        'accuracy': {
            'top1': top1,
            'epoch': epoch
        }
    }
    
    with open(f'performance_report_epoch_{epoch}.json', 'w') as f:
        json.dump(performance_report, f, indent=2)