from models.ECA_UNet import UNet
from torch.cuda.amp import autocast
import torch


def build_model(in_channels=3, out_channels=3):
    return UNet(in_channels, out_channels)


if __name__ == '__main__':
    model = build_model(3, 3).cuda()
    hazy = torch.randn(1, 3, 512, 512).cuda()
    with autocast():
        output = model(hazy)
    print(output.shape)
