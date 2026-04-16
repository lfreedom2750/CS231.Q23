import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import get_dataloaders
from src.models.vgg16_model import build_vgg16
from src.evaluation.evaluate import evaluate_model, print_metrics


def format_epoch_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes}m {secs:.0f}s"
    return f"{secs:.0f}s"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc


def main():
    data_dir = "data/splits"
    batch_size = 16
    image_size = 224
    num_epochs = 10
    learning_rate = 1e-4
    num_workers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers
    )

    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Class to index:", class_to_idx)

    model = build_vgg16(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    save_dir = Path("experiments/vgg16_scratch")
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_model_path = save_dir / "best_vgg16_scratch.pth"

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes
        )

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = format_epoch_time(epoch_time)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(
            f"{epoch_time_str} - "
            f"accuracy: {train_acc:.4f} - "
            f"loss: {train_loss:.4f} - "
            f"val_accuracy: {val_metrics['accuracy']:.4f} - "
            f"val_loss: {val_metrics['loss']:.4f}"
        )

        print("Confusion Matrix (Validation):")
        print(val_metrics["confusion_matrix"])
        print("-" * 60)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_model_path)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes
    )

    print_metrics(test_metrics, class_names=class_names, split_name="Test")


if __name__ == "__main__":
    main()