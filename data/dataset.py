import os
import shutil
import yaml
from pathlib import Path
from typing import Optional, Callable

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


cfg = load_config()
IMG_SIZE = cfg["data"]["image_size"]


class VintageCarDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_names: Optional[list] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform or transforms.ToTensor()
        self.class_names = class_names

        self.image_paths = []
        self.labels = []
        self._class_to_idx = {}

        if os.path.isdir(root_dir):
            class_dirs = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])
            if class_names is not None:
                class_dirs = [c for c in class_dirs if c in class_names]
            self._class_to_idx = {name: idx for idx, name in enumerate(class_dirs)}
            for class_name in class_dirs:
                class_path = os.path.join(root_dir, class_name)
                for fname in os.listdir(class_path):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        self.image_paths.append(os.path.join(class_path, fname))
                        self.labels.append(self._class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_to_idx(self):
        return self._class_to_idx


def get_transforms(augment: bool = True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=cfg["augmentation"]["random_resized_crop_scale"]),
            transforms.RandomHorizontalFlip(p=cfg["augmentation"]["horizontal_flip_prob"]),
            transforms.ColorJitter(
                brightness=cfg["augmentation"]["color_jitter_brightness"],
                contrast=cfg["augmentation"]["color_jitter_contrast"],
                saturation=cfg["augmentation"]["color_jitter_saturation"],
            ),
            transforms.RandomRotation(cfg["augmentation"]["rotation_degrees"]),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(
        train_dir, transform=get_transforms(augment=True)
    )
    val_dataset = datasets.ImageFolder(
        val_dir, transform=get_transforms(augment=False)
    )
    test_dataset = datasets.ImageFolder(
        test_dir, transform=get_transforms(augment=False)
    )

    import torch
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes
