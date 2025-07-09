# 创建一个专门的镜像配置脚本
mirror_setup = '''#!/bin/bash

echo "🔧 配置Qwen模型专用镜像源..."

# 设置Hugging Face镜像
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

# 设置缓存目录
export HUGGINGFACE_HUB_CACHE=/home/stu_b/.cache/huggingface
echo "export HUGGINGFACE_HUB_CACHE=/home/stu_b/.cache/huggingface" >> ~/.bashrc

# 创建缓存目录
mkdir -p /home/stu_b/.cache/huggingface

# 设置pip镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo "✅ 镜像源配置完成！"
echo "🔄 请运行以下命令使配置生效："
echo "source ~/.bashrc"
echo ""
echo "然后运行Qwen专用程序："
echo "cd /home/stu_b/BIKE"
echo "python qwen_only_lmdb_generator.py"
'''

with open('/home/stu_b/BIKE/qwen/setup_qwen_mirrors.sh', 'w', encoding='utf-8') as f:
    f.write(mirror_setup)

print("已创建Qwen专用镜像配置脚本")
print("文件路径: /home/user/setup_qwen_mirrors.sh")
