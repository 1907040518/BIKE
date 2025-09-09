
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

class CustomQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """
    自定义的Qwen2-VL模型，支持外部特征注入
    """
    
    def __init__(self, config):
        super().__init__(config)
        # 添加特征投影层，将CLIP特征映射到Qwen的hidden_size
        self.clip_feature_projector = nn.Linear(512, config.hidden_size)  # 512是CLIP的特征维度
        # 🔧 确保投影层使用正确的数据类型和设备
        if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
            self.clip_feature_projector = self.clip_feature_projector.to(dtype=config.torch_dtype)


    def prepare_inputs_for_multimodal(self, input_ids, position_ids, attention_mask, 
                                    past_key_values, labels, clip_features=None, **kwargs):
        """
        准备多模态输入，将CLIP特征集成到输入嵌入中
        """
        if clip_features is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # 获取模型的主设备和数据类型
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype  # 🔧 获取模型的数据类型
        print(f"🔧 模型主设备: {model_device}, 数据类型: {model_dtype}")
        
        # 检查投影层当前设备
        projector_device = next(self.clip_feature_projector.parameters()).device
        print(f"🔧 投影层当前设备: {projector_device}")
        
        # 强制移动投影层到主设备
        if projector_device != model_device:
            print(f"🔧 移动投影层从 {projector_device} 到 {model_device}")
            self.clip_feature_projector = self.clip_feature_projector.to(model_device)
        
        # 🔧 确保投影层使用正确的数据类型
        if self.clip_feature_projector.weight.dtype != model_dtype:
            print(f"🔧 投影层数据类型转换: {self.clip_feature_projector.weight.dtype} -> {model_dtype}")
            self.clip_feature_projector = self.clip_feature_projector.to(dtype=model_dtype)
        
        # 确保所有输入都在同一设备上并使用正确的数据类型
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        if position_ids is not None:
            position_ids = position_ids.to(model_device)
        
        # 🔧 确保CLIP特征使用模型的数据类型
        clip_features = clip_features.to(device=model_device, dtype=model_dtype)
        print(f"🔧 CLIP特征设备: {clip_features.device}, 类型: {clip_features.dtype}")
        
        # 确保CLIP特征有正确的维度
        if clip_features.dim() == 2:
            clip_features = clip_features.unsqueeze(1)
            print(f"🔧 CLIP特征维度扩展: {clip_features.shape}")
        
        # 将CLIP特征投影到Qwen的特征空间
        projected_features = self.clip_feature_projector(clip_features)
        print(f"🔧 投影特征设备: {projected_features.device}, 形状: {projected_features.shape}, 类型: {projected_features.dtype}")
        
        # 获取输入嵌入
        inputs_embeds = self.get_input_embeddings()(input_ids)
        print(f"🔧 输入嵌入设备: {inputs_embeds.device}, 形状: {inputs_embeds.shape}, 类型: {inputs_embeds.dtype}")
        
        # 🔧 确保投影特征与输入嵌入数据类型一致
        if projected_features.dtype != inputs_embeds.dtype:
            projected_features = projected_features.to(dtype=inputs_embeds.dtype)
            print(f"🔧 投影特征类型转换为: {projected_features.dtype}")
        
        # 确保投影特征与输入嵌入维度兼容
        batch_size = inputs_embeds.shape[0]
        if projected_features.shape[0] != batch_size:
            projected_features = projected_features.expand(batch_size, -1, -1)
            print(f"🔧 投影特征batch扩展: {projected_features.shape}")
        
        # 在序列开头插入CLIP特征
        new_inputs_embeds = torch.cat([
            projected_features,  # CLIP特征在前
            inputs_embeds       # 原始文本嵌入在后
        ], dim=1)
        
        print(f"🔧 合并后嵌入设备: {new_inputs_embeds.device}, 形状: {new_inputs_embeds.shape}, 类型: {new_inputs_embeds.dtype}")
        
        # 调整attention_mask
        clip_attention = torch.ones(
            (attention_mask.shape[0], projected_features.shape[1]), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        new_attention_mask = torch.cat([clip_attention, attention_mask], dim=1)
        print(f"🔧 新attention_mask形状: {new_attention_mask.shape}")
        
        # 调整position_ids
        if position_ids is not None:
            clip_positions = torch.arange(
                projected_features.shape[1], 
                dtype=position_ids.dtype, 
                device=position_ids.device
            ).unsqueeze(0).expand(position_ids.shape[0], -1)
            position_ids = position_ids + projected_features.shape[1]
            new_position_ids = torch.cat([clip_positions, position_ids], dim=1)
            print(f"🔧 新position_ids形状: {new_position_ids.shape}")
        else:
            new_position_ids = None
        
        return None, new_position_ids, new_attention_mask, past_key_values, new_inputs_embeds, labels


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        clip_features=None,  # 新增参数
        **kwargs
    ):
        """重写forward方法以支持CLIP特征"""
        
        if inputs_embeds is None and clip_features is not None:
            # 准备多模态输入
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                clip_features
            )
        
        # 调用父类的forward方法
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs
        )

    def generate(
        self,
        input_ids=None,
        clip_features=None,  # 新增参数
        **kwargs
    ):
        """重写generate方法以支持CLIP特征"""
        
        if clip_features is not None:
            # 在生成时准备多模态输入
            attention_mask = kwargs.get('attention_mask', None)
            position_ids = kwargs.get('position_ids', None)
            
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                clip_features
            )
            
            # 更新kwargs
            kwargs['attention_mask'] = attention_mask
            kwargs['position_ids'] = position_ids
            if inputs_embeds is not None:
                kwargs['inputs_embeds'] = inputs_embeds
                input_ids = None
        
        return super().generate(input_ids=input_ids, **kwargs)


class ImprovedQwenCLIPLMDBGenerator:
    """改进的生成器类"""
    
    def __init__(self, frames_per_video=5, gpu_ids=None, clip_config=None):
        # ... 保持原有初始化代码 ...
        self.frames_per_video = frames_per_video
        self.gpu_ids = gpu_ids if gpu_ids is not None else [0]
        self.device = torch.device(f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        
        self.clip_config = clip_config or {
            'arch': 'ViT-B/32',
            'embed_dim': 512,
            'residual_layers_to_use': [0, 11],
            'mvs_layers_to_use': [0, 11]
        }
        
        self.clip_model = None
        self.qwen_model = None
        self.qwen_processor = None
        
    def load_custom_qwen_model(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        """加载自定义Qwen模型"""
        print(f"🔧 正在加载自定义Qwen模型: {model_name}")
        
        try:
            # 加载处理器
            self.qwen_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # 加载自定义模型
            self.qwen_model = CustomQwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if len(self.gpu_ids) > 1 else {"": self.gpu_ids[0]},
                trust_remote_code=True
            )
            
            self.qwen_model.eval()
            print("✅ 自定义Qwen模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 自定义Qwen模型加载失败: {e}")
            return False
    
    def generate_description_with_clip_features(
        self, 
        clip_features: torch.Tensor,  # [T, 512] CLIP特征
        prompt: str = "Analyze this video and describe the action being performed."
    ) -> str:
        """
        使用CLIP特征生成视频描述
        """
        print(f"🤖 使用CLIP特征生成视频描述...")
        
        try:
            # 1. 构建输入文本（不包含图像token，因为我们直接传递特征）
            text_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # 2. 处理文本
            text_inputs = self.qwen_processor(
                text=[text_prompt],
                images=None,
                return_tensors="pt",
                padding=True
            )
            
            # 3. 移动到设备
            input_ids = text_inputs['input_ids'].to(self.device)
            attention_mask = text_inputs['attention_mask'].to(self.device)
            clip_features = clip_features.to(self.device)
            
            # 4. 使用自定义模型生成
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    clip_features=clip_features,  # 传递CLIP特征
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.8,
                    pad_token_id=self.qwen_processor.tokenizer.eos_token_id
                )
            
            # 5. 解码输出
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[0][input_length:]
            
            output_text = self.qwen_processor.tokenizer.decode(
                new_tokens, 
                skip_special_tokens=True
            )
            
            return output_text.strip() if output_text.strip() else "生成的描述为空"
            
        except Exception as e:
            error_msg = f"❌ 描述生成失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    def process_video_with_improved_method(
        self, 
        images: List[Image.Image], 
        residuals: List[Image.Image],
        motion_vectors: List[Image.Image],
        prompt: str = "Analyze these video frames and describe the action."
    ) -> str:
        """
        改进的视频处理方法
        """
        print(f"🤖 使用改进方法处理 {len(images)} 帧视频...")
        
        try:
            # 1. 预处理数据
            image_batch, residual_batch, mv_batch = self.preprocess_multimodal_data(
                images, residuals, motion_vectors
            )
            
            # 2. 使用CLIP编码
            clip_features = self.encode_with_clip(image_batch, residual_batch, mv_batch)
            
            # 3. 聚合视频特征（可以尝试不同的聚合方式）
            # 方式1: 平均池化
            aggregated_features = clip_features.mean(dim=0, keepdim=True)  # [1, 512]
            
            # 方式2: 最大池化
            # aggregated_features = clip_features.max(dim=0, keepdim=True)[0]  # [1, 512]
            
            # 方式3: 加权平均（给中间帧更高权重）
            # weights = torch.softmax(torch.arange(len(clip_features), dtype=torch.float32), dim=0)
            # aggregated_features = (clip_features * weights.unsqueeze(1)).sum(dim=0, keepdim=True)
            
            # 4. 生成描述
            description = self.generate_description_with_clip_features(
                aggregated_features, prompt
            )
            
            return description
            
        except Exception as e:
            error_msg = f"❌ 视频处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg

# 使用示例代码
def example_usage():
    """使用示例"""
    
    # 创建改进的生成器
    generator = ImprovedQwenCLIPLMDBGenerator(
        frames_per_video=5,
        gpu_ids=[0],
        clip_config={'arch': 'ViT-B/32', 'embed_dim': 512}
    )
    
    # 初始化模型
    if generator.load_clip_model() and generator.load_custom_qwen_model():
        print("✅ 模型初始化成功")
        
        # 处理视频（假设你已经有了图像数据）
        # images, residuals, motion_vectors = extract_from_lmdb(...)
        # description = generator.process_video_with_improved_method(
        #     images, residuals, motion_vectors
        # )
        # print(f"生成的描述: {description}")
    else:
        print("❌ 模型初始化失败")


