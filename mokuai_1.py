import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积（Depthwise Convolution）
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=in_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.depthwise_relu = nn.ReLU(inplace=True)

        # 逐点卷积（Pointwise Convolution）
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.pointwise_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 深度卷积
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)

        # 逐点卷积
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_relu(x)

        return x


# 定义 MobileNetV2 的倒残差块（Inverted Residual Block）
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        # 中间扩展层
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_channels = in_channels * expand_ratio

        self.conv = nn.Sequential(
            # 1x1 卷积扩展层
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),

            # 深度卷积层
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, groups=hidden_channels,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),

            # 1x1 卷积压缩层
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)  # 如果输入与输出的形状相同，做快捷连接
        else:
            return self.conv(x)  # 否则直接返回卷积结果


# 定义 MobileNetV2 模型
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()

        # 定义网络结构
        self.model = nn.Sequential(
            # 第一层普通卷积
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            # 倒残差块（Inverted Residual Block）
            InvertedResidual(32, 16, stride=1, expand_ratio=1),
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 96, stride=2, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 160, stride=2, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 320, stride=1, expand_ratio=6),

            # 最后一个卷积层
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),

            # 全局平均池化
            nn.AdaptiveAvgPool2d(1)
        )

        # 分类层
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


# 创建 MobileNetV2 模型实例
model = MobileNetV2(num_classes=1000)

# 打印模型结构
print(model)


# Ghost Module
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup

        # 1x1 convolution for feature generation
        self.conv1 = nn.Conv2d(inp, oup // 2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(oup // 2)

        # Depthwise separable convolution for feature transformation
        self.conv2 = nn.Conv2d(oup // 2, oup // 2, kernel_size, stride, kernel_size // 2, groups=oup // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(oup // 2)

        # Pointwise convolution for mixing the generated features
        self.conv3 = nn.Conv2d(oup // 2, oup // 2, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup // 2)

        # Activation function (ReLU)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        # First, perform 1x1 convolution to generate new features
        x1 = self.relu(self.bn1(self.conv1(x)))

        # Then, perform depthwise separable convolution
        x2 = self.relu(self.bn2(self.conv2(x1)))

        # Mix the features together
        x3 = self.bn3(self.conv3(x2))

        # Concatenate the two parts to form the final output
        return torch.cat([x1, x3], 1)


# GhostNet Model
class GhostNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GhostNet, self).__init__()

        # Initial Conv Layer
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # GhostNet Layers (Ghost modules + Depthwise Separable Conv)
        self.layers = nn.Sequential(
            GhostModule(16, 32, kernel_size=3, stride=2),  # Stage 1
            GhostModule(32, 64, kernel_size=3, stride=2),  # Stage 2
            GhostModule(64, 128, kernel_size=3, stride=2),  # Stage 3
            GhostModule(128, 256, kernel_size=3, stride=2),  # Stage 4
            GhostModule(256, 512, kernel_size=3, stride=2),  # Stage 5
        )

        # Global average pooling and fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x