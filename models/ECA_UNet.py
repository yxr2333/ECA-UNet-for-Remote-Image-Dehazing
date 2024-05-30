import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ECAModule import ECAModule
from torch.cuda.amp import autocast


class DoubleConv(nn.Module):
    """
    双卷积模块，包括两个卷积层和激活函数ReLU
    在最后一个卷积层后添加了ECA模块
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ECAModule(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样模块，包括最大池化和双卷积
    最大池化用于减少特征图尺寸
    双卷积用于提取特征
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 定义下采样模块，包括最大池化和双卷积，使用最大池化来使特征图尺寸减半，使用双卷积来提取特征
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样模块
    通过上采样将特征图尺寸恢复到原始尺寸
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # 根据bilinear参数选择上采样方式（默认为双线性插值）
        # 如果bilinear为True，则使用双线性插值上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 使用卷积来减少合并后的通道数
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)
        # 否则使用转置卷积上采样
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 对x1完成上采样
        x1 = self.up(x1)
        # 计算x1和x2在高和宽上的尺度差异，然后对x1进行填充以匹配x2的尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 将x1和x2在通道维度上合并
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    输出卷积层，将最终的特征图转换为最终的输出
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet模型的Pytorch实现，包括编码器和解码器
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        # 输入卷积层，将输入图片的通道数转换为64
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        # 将特征图依次传入编码器和解码器
        # 经过连续的三个下采样
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # 经过连续的三个上采样
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # 最后通过输出卷积层得到最终的输出
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=3).cuda()
    hazy = torch.randn(1, 3, 256, 256).cuda()
    with autocast():
        output = model(hazy)
    print(output.shape)
