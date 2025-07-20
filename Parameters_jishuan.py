import torch
import torch.nn as nn
from thop import profile
import torchvision.models as models
from yanshi_model import yanshi_resnet
from yanshi_model_xiaorong import yanshi_xiaorong
from yanshi_model_xiaorong_zhuyili import SE_ResNet2262
from yanshi_model_xiaorong_CABM import CBAM_ResNet2262_CABM

#model = models.resnet101()
model = CBAM_ResNet2262_CABM()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params / 1e6:.6f}M")  # 转换成M单位