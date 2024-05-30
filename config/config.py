import argparse

parser = argparse.ArgumentParser(description='参数介绍')

parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_epochs', type=int, default=120, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--use_sar', type=bool, default=True, help='use sar image')
parser.add_argument('--haze1k_save_path', type=str, default='weights/haze1k/ECA_UNet.pth', help='model saved path')
parser.add_argument('--haze1k_sar_save_path', type=str, default='weights/haze1k/DehazeNet.pth', help='model saved path')
parser.add_argument('--reside_save_path', type=str, default='weights/reside/ECA_UNet.pth', help='model saved path')
parser.add_argument('--img_size', type=int, default=512, help='image size')

args = parser.parse_args()

