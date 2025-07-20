import torch
from thop import profile
from yanshi_model import yanshi_resnet
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
from torch import nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torchvision.models as models
from net_checkpoint import save_checkpoint, load_checkpoint
from yanshi_model_xiaorong import yanshi_xiaorong

model = models.resnet50
model = yanshi_resnet()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
dummy_input = torch.randn(16, 3,224,224, dtype=torch.float).to(device)
repetitions=100
total_time = 0
with torch.no_grad():
  for rep in range(repetitions):
     starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     starter.record()
     _ = model(dummy_input)
     ender.record()
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)/1000
     total_time += curr_time
Throughput = (repetitions*16)/total_time
print('Final Throughput:',Throughput)












