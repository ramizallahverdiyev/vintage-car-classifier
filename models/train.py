import os
import sys
import time
import json
import yaml

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_config, create_dataloaders
from models.model import build_model, get_device


cfg = load_config()
device = get_device()
print(f"Using device: {device}")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_desc=""):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1:
            pct = (batch_idx + 1) / n_batches * 100
            print(f"  {epoch_desc} [{batch_idx+1}/{n_batches}] {pct:.0f}%", end="\r", flush=True)

    print(" " * 60, end="\r", flush=True)
    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def main():
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=cfg["paths"]["logs_dir"])

    data_dir = cfg["data"]["processed_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    patience = cfg["training"]["early_stopping_patience"]

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    model = build_model(
        num_classes=num_classes,
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_acc = 0.0
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        epoch_desc = f"Epoch {epoch}/{epochs}"
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch_desc)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - train_start

        avg_epoch_time = total_elapsed / epoch
        eta = avg_epoch_time * (epochs - epoch)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        def fmt_seconds(s):
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            if h > 0:
                return f"{h}h{m}m"
            if m > 0:
                return f"{m}m{sec}s"
            return f"{sec}s"

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.3f} / {train_acc:.1f}% | "
            f"Val:   {val_loss:.3f} / {val_acc:.1f}% | "
            f"{fmt_seconds(epoch_elapsed)} | "
            f"ETA: {fmt_seconds(eta)}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            epochs_no_improve = 0
            checkpoint_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_names": class_names,
            }, checkpoint_path)
            print(f"  -> Saved checkpoint ({val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Save metadata
    meta = {
        "class_names": class_names,
        "num_classes": num_classes,
        "best_val_acc": round(best_acc, 2),
        "test_acc": round(test_acc, 2),
        "device": str(device),
    }
    meta_path = os.path.join(cfg["paths"]["checkpoint_dir"], "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
