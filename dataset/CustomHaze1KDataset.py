import os

import torchvision.transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import Dataset


class CustomHaze1KDataset(Dataset):
    def __init__(self, root_dir, type="train", fog_level="Haze1k_thin", transform=None, include_sar=False):
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
        self.include_sar = include_sar

        self.data_path = os.path.join(root_dir, fog_level, "dataset", type)
        self.inputs_dir = os.path.join(self.data_path, "input")
        self.targets_dir = os.path.join(self.data_path, "target")
        self.image_filenames = [f for f in os.listdir(self.inputs_dir) if
                                f.endswith("-inputs.jpg") or f.endswith("-inputs.png")]

        to_sar_type = {"train": "train", "valid": "val", "test": "test"}
        if self.include_sar:
            self.sar_dir = os.path.join(root_dir, fog_level, "SAR", to_sar_type[type])
            self.sar_filenames = [f for f in os.listdir(self.sar_dir) if f.endswith(".jpg") or f.endswith(".png")]

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

        if self.include_sar:
            # 从文件名中提取编号并计算对应的SAR图像名称
            img_num = int(img_name.split('-')[0])

            if self.type == "test":
                if self.fog_level in ['Haze1k_thin', 'Haze1k_thick']:
                    sar_num = img_num - 355  # test中的356号图片对应于SAR中的1号图片
                elif self.fog_level == "Haze1k_moderate":
                    sar_num = img_num
            elif self.type == "valid":
                if self.fog_level in ["Haze1k_thin", "Haze1k_thick"]:
                    sar_num = img_num - 320  # valid中的321号图片对应于SAR中的1号图片
                elif self.fog_level == "Haze1k_moderate":
                    sar_num = img_num  # moderate中的valid从1开始
                else:
                    raise ValueError(f"Unknown fog level: {self.fog_level}")
            elif self.type == "train":
                sar_num = img_num  # 根据实际情况修改
            else:
                raise ValueError(f"Unknown dataset type: {self.type}")

            # SAR图像名称可能是.jpg或.png
            sar_name_jpg = f"{sar_num}.jpg"
            sar_name_png = f"{sar_num}.png"
            sar_path_jpg = os.path.join(self.sar_dir, sar_name_jpg)
            sar_path_png = os.path.join(self.sar_dir, sar_name_png)

            sar_image = None
            try:
                if os.path.exists(sar_path_jpg):
                    sar_image = Image.open(sar_path_jpg).convert("RGB")
                elif os.path.exists(sar_path_png):
                    sar_image = Image.open(sar_path_png).convert("RGB")
                else:
                    raise FileNotFoundError(f"File {sar_name_jpg} or {sar_name_png} does not exist.")

                if self.transform and sar_image:
                    sar_image = self.transform(sar_image)
            except OSError:
                print(f"Skipping corrupted SAR image: {sar_path_jpg} or {sar_path_png}")
                sar_image = None

            return image, target, sar_image

        return image, target


def get_combined_dataloader(transform, root_dir, batch_size, include_sar=False):
    train_datasets = []
    valid_datasets = []
    fog_levels = ["Haze1k_thin", "Haze1k_moderate", "Haze1k_thick"]
    for fog_level in fog_levels:
        train_dataset = CustomHaze1KDataset(root_dir=root_dir, fog_level=fog_level, type="train",
                                            transform=transform, include_sar=include_sar)
        valid_dataset = CustomHaze1KDataset(root_dir=root_dir, fog_level=fog_level, type="valid",
                                            transform=transform, include_sar=include_sar)
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
                                                                                       batch_size=1,
                                                                                       include_sar=True)
    print(f'Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}')
