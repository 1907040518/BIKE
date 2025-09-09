
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
        
    def prepare_inputs_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels,
        clip_features=None,  # 新增参数：CLIP编码的特征
        **kwargs
    ):
        """
        借鉴LLaVA的多模态输入准备方法
        """
        # 如果没有CLIP特征或者是生成过程中的单token情况，直接返回
        if clip_features is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # 将CLIP特征投影到Qwen的特征空间
        projected_features = self.clip_feature_projector(clip_features)  # [T, hidden_size]
        
        # 获取文本嵌入
        inputs_embeds = self.get_input_embeddings()(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # 找到图像占位符的位置（假设使用特殊token）
        # 这里简化处理：将CLIP特征插入到序列开头
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        num_clip_tokens = projected_features.shape[0]
        
        # 重新组织嵌入：[BOS] + [CLIP_FEATURES] + [剩余文本]
        new_inputs_embeds = torch.cat([
            inputs_embeds[:, :1, :],  # BOS token
            projected_features.unsqueeze(0),  # CLIP特征 [1, T, hidden_size]
            inputs_embeds[:, 1:, :]   # 剩余文本tokens
        ], dim=1)
        
        # 更新attention_mask
        if attention_mask is not None:
            clip_attention = torch.ones(batch_size, num_clip_tokens, 
                                      dtype=attention_mask.dtype, 
                                      device=attention_mask.device)
            new_attention_mask = torch.cat([
                attention_mask[:, :1],  # BOS
                clip_attention,         # CLIP tokens
                attention_mask[:, 1:]   # 剩余文本
            ], dim=1)
        else:
            new_attention_mask = None
        
        # 更新position_ids
        if position_ids is not None:
            new_seq_len = new_inputs_embeds.shape[1]
            new_position_ids = torch.arange(new_seq_len, dtype=position_ids.dtype, 
                                          device=position_ids.device).unsqueeze(0)
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

print("✅ 代码重构完成！")
