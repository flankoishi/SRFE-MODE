import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader,Dataset
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
import shutil
from yanshi_train import yanshi_train
import warnings
warnings.filterwarnings("ignore")
import torchvision.models as models

pretrained_model = models.resnet101(models.ResNet101_Weights.DEFAULT)

##对图片进行增广,并固定位1000X1000的图片,并经行归一化
color_aug = transforms.ColorJitter(0.3, 0.2, 0.5, 0.08)
shape_aug = transforms.RandomResizedCrop(256, scale=(0.7, 1), ratio=(0.5, 2))
train_aug = transforms.Compose([color_aug, shape_aug, transforms.ToTensor(),
                               transforms.Normalize(mean=[0.6147, 0.5833, 0.5353],std=[0.1549,  0.1567,  0.1571])])
##对测试图片进行维度转换，转成pytorch框架的格式,resize从长方形改为正方形，
test_aug = transforms.Compose([torchvision.transforms.Resize(256),
           transforms.ToTensor(),
           transforms.Normalize(
               mean=[0.6147, 0.5833, 0.5353],std=[0.1549,  0.1567,  0.1571])])

#创建数据集
train_dataset = torchvision.datasets.ImageFolder(
    'D:\yangben\Train_ds', transform=train_aug)
test_dataset = torchvision.datasets.ImageFolder('D:\yangben\Test_ds', transform=test_aug)

#训练量
train_num = len(train_dataset)

#加载框架数据集函数，batch_size为批量数，shuffle为是否随机
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=10, shuffle=True, num_workers=8)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10, shuffle=False, num_workers=8)

#使用显卡跑
device_g = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#定义模型
yanshi_net = torchvision.models.resnet101(pretrained=False)
yanshi_net.fc = nn.Linear(yanshi_net.fc.in_features,out_features=5)
yanshi_net.to(device_g)
nn.init.xavier_uniform_(yanshi_net.fc.weight)#初始化权重
loss_func = nn.CrossEntropyLoss()##损失函数
optimizer_fun = torch.optim.Adam(yanshi_net.parameters(), lr=0.0001)##定义优化算法,lr学习率可调

tg_save_path = 'D:\yangben\weight\Yanshi_100.pth'
#torch.save(yanshi_net.state_dict(), 'save.pt')##保存训练好的权重

#开始训练
#yanshi_train(yanshi_net, 5, train_loader, test_loader, device_g, loss_func)


if __name__ == '__main__':
    yanshi_train(yanshi_net, 15, train_loader, test_loader, device_g, loss_func)






