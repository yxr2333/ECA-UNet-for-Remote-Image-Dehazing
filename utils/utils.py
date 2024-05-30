import numpy as np
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def calculate_psnr_ssim(dehazed_image, clear_image):
    # 将PIL图像转换为float numpy数组
    dehazed_image = np.array(dehazed_image).astype('float32') / 255
    clear_image = np.array(clear_image).astype('float32') / 255

    # 计算PSNR和SSIM
    psnr = compare_psnr(clear_image, dehazed_image)
    ssim = compare_ssim(clear_image, dehazed_image, multichannel=True, win_size=3,data_range=1)

    return psnr, ssim
