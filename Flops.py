import torch
from thop import profile
import torchvision.models as models
from yanshi_model import yanshi_resnet
from yanshi_model_xiaorong import yanshi_xiaorong


#model = models.resnet152()
model = yanshi_xiaorong()
input1 = torch.randn(4, 3, 224, 224)
flops, params = profile(model, inputs=(input1, ))
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
