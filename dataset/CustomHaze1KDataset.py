import os

import torchvision.transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import Dataset


class CustomHaze1KDataset(Dataset):
    def __init__(self, root_dir, type="train", fog_level="Haze1k_thin", transform=None):
        """
        root_dir: 数据集的根目录路径
        type: 数据集的类型（train, valid, test）
        fog_level: 雾的类型（Haze1k_thin, Haze1k_moderate, Haze1k_thick）
        transform: 应用于数据的转换操作
        include_sar: 是否包含SAR数据
        """
        self.root_dir = root_dir
        self.type = type
        self.fog_level = fog_level
        self.transform = transform

        self.data_path = os.path.join(root_dir, fog_level, "dataset", type)
        self.inputs_dir = os.path.join(self.data_path, "input")
        self.targets_dir = os.path.join(self.data_path, "target")
        self.image_filenames = [f for f in os.listdir(self.inputs_dir) if
                                f.endswith("-inputs.jpg") or f.endswith("-inputs.png")]

        to_sar_type = {"train": "train", "valid": "val", "test": "test"}

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        input_path = os.path.join(self.inputs_dir, img_name)
        target_path = os.path.join(self.targets_dir, img_name.replace("inputs", "targets"))

        try:
            image = Image.open(input_path).convert("RGB")
            target = Image.open(target_path).convert("RGB")
        except OSError:
            print(f"Skipping corrupted image: {input_path}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target


def get_combined_dataloader(transform, root_dir, batch_size):
    train_datasets = []
    valid_datasets = []
    fog_levels = ["Haze1k_thin", "Haze1k_moderate", "Haze1k_thick"]
    for fog_level in fog_levels:
        train_dataset = CustomHaze1KDataset(root_dir=root_dir, fog_level=fog_level, type="train",
                                            transform=transform)
        valid_dataset = CustomHaze1KDataset(root_dir=root_dir, fog_level=fog_level, type="valid",
                                            transform=transform)
        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)
    # 合并数据集
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_valid_dataset = ConcatDataset(valid_datasets)
    # 创建数据加载器
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(concat_valid_dataset, batch_size=batch_size, shuffle=False)
    return concat_train_dataset, concat_valid_dataset, train_loader, valid_loader


if __name__ == "__main__":
    transform = torchvision.transforms.ToTensor()
    train_dataset, valid_dataset, train_loader, valid_loader = get_combined_dataloader(transform=transform,
                                                                                       root_dir='./Haze1k',
                                                                                       batch_size=1)
    print(f'Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}')
