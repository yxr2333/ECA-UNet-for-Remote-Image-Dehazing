import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from dataset.CustomHaze1KDataset import get_combined_dataloader
from models.model import build_model
from utils.utils import calculate_psnr_ssim
from config.config import args
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import os

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 加速训练
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

print("Loading dataset...")
train_dataset, valid_dataset, train_loader, valid_loader = get_combined_dataloader(transform,
                                                                                   root_dir="./dataset/Haze1k",
                                                                                   batch_size=args.batch_size)
print(f'Load dataset successfully, Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}')

print(f'Model save path: {args.haze1k_save_path}')
os.makedirs(args.haze1k_save_path, exist_ok=True)

print("Loading model and other parameters...")
model = build_model(3, 3).to(args.device)
criterion = torch.nn.MSELoss().to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
scaler = GradScaler(device='cuda')
best_valid_loss = float('inf')
try:
    state = torch.load(f'{args.haze1k_save_path}/last.pth', weights_only=True)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    best_valid_loss = state['best_valid_loss']
    print(f"Model loaded successfully, best valid loss: {best_valid_loss}")
except:
    print("Model loaded failed, train from scratch")


# 修改train_model函数
def train_model(model, data_loader, criterion, optimizer, device='cuda'):
    # 合并所有雾类型的训练和验证数据

    model.train()
    total_loss = 0
    # for inputs, targets in data_loader:
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs_dehaze = model(inputs)
            loss = criterion(outputs_dehaze, targets)

        # loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(data_loader)
    return avg_train_loss


def validate_model(model, data_loader, criterion, device='cuda'):
    model.eval()
    val_loss, total_psnr, total_ssim = 0, 0, 0
    with torch.no_grad():

        # for inputs, targets in data_loader:
        for inputs, targets in tqdm(valid_loader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)

            # 计算总损失
            with autocast('cuda'):
                outputs_dehaze = model(inputs)
                loss = criterion(outputs_dehaze, targets)

            val_loss += loss.item()

            # Calculate PSNR and SSIM
            outputs_gpu = outputs_dehaze.detach()
            targets_gpu = targets.detach()

            for output, target in zip(outputs_gpu, targets_gpu):
                output_pil = to_pil_image(output).convert("RGB")
                target_pil = to_pil_image(target).convert("RGB")

                psnr, ssim = calculate_psnr_ssim(output_pil, target_pil)
                total_psnr += psnr
                total_ssim += ssim

        avg_valid_loss = val_loss / len(data_loader)

        avg_psnr = total_psnr / len(data_loader.dataset)
        avg_ssim = total_ssim / len(data_loader.dataset)

        return avg_valid_loss, avg_psnr, avg_ssim


if __name__ == '__main__':
    print(f'Using device: {args.device}, Num epochs: {args.num_epochs}')
    print("Start training...")
    best_save_path = f'{args.haze1k_save_path}/best.pth'
    last_save_path = f'{args.haze1k_save_path}/last.pth'
    for epoch in range(args.num_epochs):
        train_loss = train_model(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=args.device)
        valid_loss, valid_psnr, valid_ssim = validate_model(
            model=model,
            data_loader=valid_loader,
            criterion=criterion,
            device=args.device)

        print(f"Epoch {epoch + 1}/{args.num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
              f"Valid PSNR: {valid_psnr:.2f}, Valid SSIM: {valid_ssim:.3f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_valid_loss': best_valid_loss
            }, best_save_path)
            print("Best model saved successfully")

        scheduler.step()
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_valid_loss': best_valid_loss
        }, last_save_path)

    print("Training finished")
