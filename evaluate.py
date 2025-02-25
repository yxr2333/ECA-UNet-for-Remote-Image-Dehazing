import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config.config import args
from models.ECA_UNet import UNet
from utils.utils import calculate_psnr_ssim

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def evaluate(model, test_dir="Haze1k/Haze1k_moderate/dataset/test"):
    input_dir = os.path.join(test_dir, "input")
    target_dir = os.path.join(test_dir, "target")

    psnr_list = []
    ssim_list = []

    for i, image_name in enumerate(os.listdir(input_dir)):

        if not image_name.endswith("-inputs.png"):
            continue
        fig, axes = plt.subplots(1, 3)
        # 加载并预处理图像
        input_image_path = os.path.join(input_dir, image_name)
        target_image_path = os.path.join(target_dir, image_name.replace("inputs", "targets"))

        input_image = Image.open(input_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB").resize((512, 512))

        axes[0].imshow(input_image)
        axes[0].axis('off')
        axes[0].set_title('Test ' + str(i) + ' Input')

        axes[1].imshow(target_image)
        axes[1].axis('off')
        axes[1].set_title('Test ' + str(i) + ' Target')

        input_tensor = transform(input_image).unsqueeze(0).cuda()  # 添加批次维度

        # 模型去雾
        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_tensor = output_tensor.squeeze(0)
        output_tensor = output_tensor.clamp(0, 1)

        target_image = target_image.resize((512, 512))
        output_image = transforms.ToPILImage()(output_tensor)  # 移除批次维度

        axes[2].imshow(output_image)
        axes[2].axis('off')
        axes[2].set_title('Test ' + str(i) + ' Output')

        plt.subplots_adjust(wspace=0.2)
        if i % 20 == 0:
            plt.show()
        else:
            plt.close('all')

        # 计算PSNR和SSIM
        psnr, ssim = calculate_psnr_ssim(output_image, target_image)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    # 计算平均指标
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    return avg_psnr, avg_ssim


# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = f'{args.haze1k_save_path}/last.pth'  # 更新为你的模型路径

model = UNet(in_channels=3, out_channels=3).to(device)  # 确保这里的参数与训练时一致
# model = torch.nn.DataParallel(model, device_ids=[0])

checkpoint = torch.load(model_path)

model.load_state_dict(checkpoint['model'])

model.eval()  # 切换到评估模式

# 评估模型
avg_psnr_thin, avg_ssim_thin = evaluate(model, test_dir="dataset/Haze1k/Haze1k_thin/dataset/test")

import concurrent.futures


def evaluate_model_on_testset(model, test_dir):
    return evaluate(model, test_dir)


# 主函数或其他适当位置
if __name__ == "__main__":
    # 创建一个线程池执行器
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 定义要评估的数据集目录
        test_dirs = {
            "thin": "dataset/Haze1k/Haze1k_thin/dataset/test",
            "moderate": "dataset/Haze1k/Haze1k_moderate/dataset/test",
            "thick": "dataset/Haze1k/Haze1k_thick/dataset/test"
        }

        # 使用future字典来跟踪每个调用
        futures = {
            haze_density: executor.submit(evaluate_model_on_testset, model, test_dir)
            for haze_density, test_dir in test_dirs.items()
        }

        # 等待每个线程完成并打印结果
        for haze_density, future in futures.items():
            avg_psnr, avg_ssim = future.result()
            print(
                f'Assess on Haze1k {haze_density.capitalize()}, Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.3f}')
