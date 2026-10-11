import torch

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchvision import datasets
from torchvision import transforms


def compute_mnist_mean_std(data_dir):
    raw_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True
    )

    images = raw_train_dataset.data.float() / 255.0

    mean = images.mean().item()
    std = images.std().item()

    return mean, std


def build_transform(mean, std):

    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[mean],
            std=[std]
        )
    ])

    return transform


def create_dataloaders(
    data_dir,
    batch_size,
    train_ratio,
    seed
):
    """
    Tạo:
        train_dataloader
        valid_dataloader
        test_dataloader
    """

    # ========================================================
    # 1. COMPUTE NORMALIZATION STATS
    # ========================================================

    mean, std = compute_mnist_mean_std(
        data_dir
    )

    print(f"MNIST mean: {mean:.4f}")
    print(f"MNIST std : {std:.4f}")


    # ========================================================
    # 2. CREATE TRANSFORM
    # ========================================================

    transform = build_transform(
        mean,
        std
    )


    # ========================================================
    # 3. CREATE TRAIN DATASET
    # ========================================================

    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )


    # ========================================================
    # 4. CREATE TEST DATASET
    # ========================================================

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )


    # ========================================================
    # 5. SPLIT TRAIN / VALIDATION
    # ========================================================

    n_total = len(full_train_dataset)

    n_train = int(
        n_total * train_ratio
    )

    n_valid = (
        n_total - n_train
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_dataset, valid_dataset = random_split(
        full_train_dataset,
        [n_train, n_valid],
        generator=generator
    )


    # ========================================================
    # 6. CREATE DATALOADERS
    # ========================================================

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    return (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        mean,
        std
    )