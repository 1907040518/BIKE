import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        def custom_forward(x):
            return self.fc(x)
        return checkpoint(custom_forward, x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.transform = Transform()
        self.dtype = torch.float32

    def encode_image(self, images):
        return self.transform(images.type(self.dtype))

    def encode_text(self, text, return_token):
        # 这里简单模拟，实际根据你的代码实现
        return torch.randn(1, 10), torch.randn(1, 10)

model = Model()
image = torch.randn(1, 10)
mv = torch.randn(1, 10)
residual = torch.randn(1, 10)
text = torch.randn(1, 10)

image_feats = model.encode_image(image)
with torch.no_grad():
    mv_feats = model.encode_image(mv)
    residual_feats = model.encode_image(residual)
cls_feat, text_feats = model.encode_text(text, return_token=True)

loss = image_feats.sum()  # 假设以 image_feats 计算损失
try:
    loss.backward()
    print("没有报错，反向传播正常进行。")
except RuntimeError as e:
    print(f"报错信息: {e}")