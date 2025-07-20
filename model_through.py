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


model = models.resnet101(pretrained=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
dummy_input = torch.randn(1, 3, 224, 224,dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
   _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     starter.record()
     _ = model(dummy_input)
     ender.record()
     # WAIT FOR GPU SYNC
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)
     timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000. / mean_syn
print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
print(mean_syn)














