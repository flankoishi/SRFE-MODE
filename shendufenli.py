"""
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=False)  # 每个通道单独卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 通道混合

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
"""