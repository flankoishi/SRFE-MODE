import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from yanshi_model import yanshi_resnet
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型（以 ResNet18 为例）
#model = torchvision.models.resnet18(pretrained=True)
model = yanshi_resnet().to(device)
model.eval().to(device)

# 定义一个字典来存储每一层的特征图
feature_maps = {}


# 定义钩子函数
def hook_fn(module, input, output):
    # 获取当前层的输出特征图
    feature_maps[module] = output.detach()


# 注册钩子函数到模型的每一层卷积层、池化层、归一化层和激活层
hooks = []
for layer_name, layer in model.named_modules():
    if isinstance(layer,
                  (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.BatchNorm2d,
                   torch.nn.GELU, torch.nn.AdaptiveAvgPool2d)):
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)


# 预处理输入图像
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # 添加batch维度
    return img_tensor


# 传入一张图片并获取卷积特征图
img_tensor = preprocess_image("D:/yangben/Train_ds/Fenyanfe/1_fy_11_1.jpg")  # 这里填入你的图片路径
output = model(img_tensor)  # 执行前向传播，钩子函数会被调用并返回特征图

# 解除所有钩子，防止内存泄漏
for hook in hooks:
    hook.remove()


# 保存特征图到本地
def save_feature_map(feature_map, layer_name, save_dir='F:/feature_2'):
    # 为每个层创建文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建当前层的文件夹
    layer_save_dir = os.path.join(save_dir, layer_name)
    if not os.path.exists(layer_save_dir):
        os.makedirs(layer_save_dir)

    # 保存每个通道的特征图
    num_channels = feature_map.size(1)
    for i in range(num_channels):
        # 将每个通道的特征图保存为图像文件
        feature_image = feature_map[0, i].cpu().numpy()
        plt.imshow(feature_image, cmap='viridis')
        plt.axis('off')  # 关闭坐标轴

        # 保存图像到本地，命名为 "{layer_name}_channel_{i}.png"
        plt.savefig(os.path.join(layer_save_dir, f"{layer_name}_channel_{i}.png"))
        plt.close()  # 关闭当前图像，避免内存泄漏


# 保存每个层的特征图
def save_all_feature_maps(feature_maps, save_dir='F:/feature_2'):
    for layer, feature_map in feature_maps.items():
        # 获取层名称并保存特征图
        layer_name = str(layer).split('(')[0]  # 获取层名称（去除多余的类信息）
        save_feature_map(feature_map, layer_name, save_dir)


# 保存所有卷积层、池化层、归一化层和激活层的特征图
save_all_feature_maps(feature_maps)

print("特征图已保存到本地！")

