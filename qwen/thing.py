
# ========== 方案B：类别特定描述 + 高效融合 ==========

class ActionSpecificMultiAspectText(nn.Module):
    def __init__(self, all_actions, text_encoder):
        super().__init__()
        
        self.text_encoder = text_encoder
        
        # ========== 步骤1：为每个类别生成多方面描述 ==========
        
        print("使用LLM为每个动作生成多方面描述...")
        self.action_descriptions = self._generate_all_descriptions(all_actions)
        
        # ========== 步骤2：预编码所有描述 ==========
        
        print("预编码所有描述...")
        self.description_cache = {}
        
        with torch.no_grad():
            for action in all_actions:
                descs = self.action_descriptions[action]
                
                # 编码三个方面的描述
                self.description_cache[action] = {
                    'appearance': text_encoder(descs['appearance']),
                    'motion': text_encoder(descs['motion']),
                    'change': text_encoder(descs['change'])
                }
        
        # ========== 步骤3：轻量级融合网络 ==========
        
        # 用于融合类别名和描述的网络
        self.fusion_appearance = nn.Linear(512 * 2, 512)
        self.fusion_motion = nn.Linear(512 * 2, 512)
        self.fusion_change = nn.Linear(512 * 2, 512)
        
    def _generate_all_descriptions(self, all_actions):
        """
        使用LLM为每个动作生成多方面描述
        
        Args:
            all_actions: 所有动作类别列表
        
        Returns:
            dict: {action: {aspect: description}}
        """
        
        descriptions = {}
        
        for action in all_actions:
            # 使用LLM生成（这里简化为模板）
            descriptions[action] = {
                'appearance': self._llm_generate_appearance(action),
                'motion': self._llm_generate_motion(action),
                'change': self._llm_generate_change(action)
            }
        
        return descriptions
    
    def _llm_generate_appearance(self, action):
        """生成外观描述"""
        # 实际使用时调用LLM API
        # 这里用模板示例
        
        templates = {
            "Running": "person in athletic pose with body leaning forward",
            "Jumping": "person with legs bent and arms raised",
            "Walking": "person in upright pose with alternating leg movement",
            # ... 更多类别
        }
        
        return templates.get(action, f"person performing {action}")
    
    def _llm_generate_motion(self, action):
        """生成运动描述"""
        templates = {
            "Running": "fast horizontal motion with rhythmic alternation",
            "Jumping": "vertical upward motion with rapid acceleration",
            "Walking": "slow horizontal motion with steady pace",
            # ... 更多类别
        }
        
        return templates.get(action, f"motion pattern of {action}")
    
    def _llm_generate_change(self, action):
        """生成变化描述"""
        templates = {
            "Running": "rapid background shift and continuous texture change",
            "Jumping": "vertical scene displacement with ground separation",
            "Walking": "gradual background shift with smooth transition",
            # ... 更多类别
        }
        
        return templates.get(action, f"scene change during {action}")
    
    def forward(self, action_label):
        """
        Args:
            action_label: 类别名称
        
        Returns:
            text_app, text_motion, text_change
        """
        
        # 步骤1：编码类别名称（只调用一次text_encoder）
        text_base = self.text_encoder(action_label)  # (512,)
        
        # 步骤2：从缓存获取预编码的描述（无额外计算）
        cached_descs = self.description_cache[action_label]
        desc_app = cached_descs['appearance']    # (512,)
        desc_motion = cached_descs['motion']     # (512,)
        desc_change = cached_descs['change']     # (512,)
        
        # 步骤3：融合类别名和描述（轻量级计算）
        # 拼接后通过Linear层融合
        text_app = self.fusion_appearance(
            torch.cat([text_base, desc_app], dim=-1)
        )
        
        text_motion = self.fusion_motion(
            torch.cat([text_base, desc_motion], dim=-1)
        )
        
        text_change = self.fusion_change(
            torch.cat([text_base, desc_change], dim=-1)
        )
        
        return text_app, text_motion, text_change

# ========== 使用示例 ==========

all_actions = ["Running", "Jumping", "Walking", ...]

# 初始化（会预生成和预编码所有描述）
model = ActionSpecificMultiAspectText(all_actions, text_encoder)

# 前向传播（只需要1次text_encoder + 3次Linear）
text_app, text_motion, text_change = model("Running")

# 计算量：
# - 初始化：为每个类别编码3个描述（离线一次性）
# - 前向：1次text_encoder(类别) + 3次Linear融合
# - 非常高效！
# ========== 第一层：模态级对齐 ==========

# 输入
video = load_compressed_video()
action_label = "Running"  # 类别标签

# 步骤1：提取视觉特征
F_i = CLIP_encoder(iframe)      # (512,)
F_mv = MV_encoder(mv)           # (512,)
F_res = Res_encoder(residual)   # (512,)

# 步骤2：构建多方面的文本表示（都基于同一个类别标签）
# 方式A：使用属性锚定的提示（ATPrompt风格）
text_app = text_encoder("[V1] appearance [V2] person [V3] Running")
text_motion = text_encoder("[V1] horizontal motion [V2] fast [V3] Running")
text_change = text_encoder("[V1] texture variation [V2] shift [V3] Running")


# 步骤3：分别计算相似度
score_app = cosine_similarity(F_i, text_app)
score_mv = cosine_similarity(F_mv, text_motion)
score_res = cosine_similarity(F_res, text_change)

# 步骤4：融合模态级得分
score_modality = w1 * score_app + w2 * score_mv + w3 * score_res
