import os
import sys
import json
import yaml

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_config, create_dataloaders
from models.model import build_model, get_device


cfg = load_config()
device = get_device()


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    export_dir = cfg["paths"]["export_dir"]
    os.makedirs(export_dir, exist_ok=True)

    data_dir = cfg["data"]["processed_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    _, _, test_loader, class_names = create_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        print("Run training first: python -m models.train")
        return

    model = build_model(
        num_classes=num_classes,
        pretrained=False,
        dropout=0,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_acc: {checkpoint['val_acc']:.2f}%)")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, os.path.join(export_dir, "confusion_matrix.png"))

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_path = os.path.join(export_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Classification report saved to {report_path}")

    # Export to TorchScript
    example_input = torch.randn(1, 3, 224, 224).to(device)
    traced = torch.jit.trace(model, example_input)
    ts_path = os.path.join(export_dir, "model_torchscript.pt")
    traced.save(ts_path)
    print(f"TorchScript model exported to {ts_path}")

    # Copy metadata
    meta_src = os.path.join(checkpoint_dir, "metadata.json")
    if os.path.exists(meta_src):
        meta = json.load(open(meta_src))
        meta["test_accuracy"] = round(accuracy, 2)
        meta["class_names"] = class_names
        with open(os.path.join(export_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
