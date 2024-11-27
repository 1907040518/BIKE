"""Model definition."""
import sys
import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到 sys.path
sys.path.append(project_root)

# 导入 vim 文件夹中的 models_mamba
from Vim_main.vim import models_mamba
import torch

from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation, 
                 base_model='resnet152'):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments

        print(("""
Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):
        model_state = torch.load(r'vim_t_midclstok_ft_78p3acc.pth')
        self.base_model.load_state_dict(model_state,strict=False)
        # 根据需要的类别替换head
        feature_dim = getattr(self.base_model, 'head').in_features
        setattr(self.base_model, 'head', nn.Linear(feature_dim, num_class))

        if self._representation == 'mv':
            setattr(self.base_model, 'conv1',
                    nn.Conv2d(2, 64, 
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False))
            self.data_bn = nn.BatchNorm2d(2)
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)


    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=False)

            self._input_size = 224
        
        elif 'mamba' in base_model:
            print('Mamba')
            self.base_model=models_mamba.vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()
            self._input_size = 224

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, input):
        input = input.view((-1, ) + input.size()[-3:])
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)

        base_out = self.base_model(input)
        return base_out

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))])
