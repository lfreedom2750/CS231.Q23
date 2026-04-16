from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return train_transform, eval_transform


def get_datasets(data_dir: str, image_size: int = 224):
    data_dir = Path(data_dir)
    train_transform, eval_transform = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(
        root=data_dir / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=data_dir / "val",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=data_dir / "test",
        transform=eval_transform
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 2
):
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir, image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_dataset.classes, train_dataset.class_to_idx