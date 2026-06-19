"""
Prepare Stanford Cars dataset:
1. Read annotations from .mat files
2. Organize images into class folders
3. Split into train/val/test
"""
import os
import sys
import shutil
import random
from collections import defaultdict

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KAGGLE_CACHE = os.path.join(
    os.environ.get("HOME", os.environ.get("USERPROFILE", "")),
    ".cache", "kagglehub", "datasets", "eduardo4jesus",
    "stanford-cars-dataset", "versions", "1",
)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
SEED = 42


def load_class_names():
    meta_path = os.path.join(KAGGLE_CACHE, "car_devkit", "devkit", "cars_meta.mat")
    meta = loadmat(meta_path)
    class_names = [name[0] for name in meta["class_names"][0]]
    return class_names


def load_train_annotations():
    ann_path = os.path.join(KAGGLE_CACHE, "car_devkit", "devkit", "cars_train_annos.mat")
    ann = loadmat(ann_path)
    annotations = ann["annotations"][0]
    results = []
    for item in annotations:
        filename = str(item["fname"][0])
        class_idx = int(item["class"][0, 0]) - 1
        results.append((filename, class_idx))
    return results


def load_test_predictions():
    preds_path = os.path.join(KAGGLE_CACHE, "car_devkit", "devkit", "train_perfect_preds.txt")
    with open(preds_path) as f:
        lines = f.read().strip().splitlines()
    return [int(line.strip()) - 1 for line in lines]


def get_image_source(filename):
    paths_to_check = [
        os.path.join(KAGGLE_CACHE, "cars_train", filename),
        os.path.join(KAGGLE_CACHE, "cars_train", "cars_train", filename),
        os.path.join(KAGGLE_CACHE, "cars_test", filename),
        os.path.join(KAGGLE_CACHE, "cars_test", "cars_test", filename),
        os.path.join(KAGGLE_CACHE, "cars_train", "cars_train", filename),
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    return None


def organize_dataset():
    class_names = load_class_names()
    print(f"Total classes: {len(class_names)}")

    train_ann = load_train_annotations()

    test_img_dir = os.path.join(KAGGLE_CACHE, "cars_test", "cars_test")
    test_fnames = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".jpg")])
    test_labels = load_test_predictions()

    test_ann = list(zip(test_fnames, test_labels[:len(test_fnames)]))

    all_annotations = train_ann + test_ann
    print(f"Total annotated images: {len(all_annotations)}")

    image_map = defaultdict(list)
    for fname, class_idx in all_annotations:
        image_map[class_idx].append(fname)

    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    os.makedirs(os.path.join(PROCESSED_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "test"), exist_ok=True)

    random.seed(SEED)

    for class_idx, filenames in image_map.items():
        class_name = class_names[class_idx]
        safe_name = class_name.replace("/", "_").replace(":", "_").replace(" ", "_")

        for split_name in ["train", "val", "test"]:
            os.makedirs(os.path.join(PROCESSED_DIR, split_name, safe_name), exist_ok=True)

        random.shuffle(filenames)
        n = len(filenames)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)

        splits = [
            ("train", filenames[:n_train]),
            ("val", filenames[n_train:n_train + n_val]),
            ("test", filenames[n_train + n_val:]),
        ]

        copied = 0
        for split_name, split_files in splits:
            target_dir = os.path.join(PROCESSED_DIR, split_name, safe_name)
            for fname in split_files:
                src = get_image_source(fname)
                if src:
                    dst = os.path.join(target_dir, os.path.basename(fname))
                    shutil.copy2(src, dst)
                    copied += 1

        print(f"  [{class_idx:3d}] {safe_name}: {copied} images")

    counts = {
        "train": len(os.listdir(os.path.join(PROCESSED_DIR, "train"))),
        "val": len(os.listdir(os.path.join(PROCESSED_DIR, "val"))),
        "test": len(os.listdir(os.path.join(PROCESSED_DIR, "test"))),
    }

    num_images = sum(
        len(os.listdir(os.path.join(PROCESSED_DIR, split, cls)))
        for split in ["train", "val", "test"]
        for cls in os.listdir(os.path.join(PROCESSED_DIR, split))
    )

    total_classes = len(image_map)
    print(f"\nDone! {total_classes} classes, {num_images} images")
    print(f"  Train: {counts['train']} classes")
    print(f"  Val:   {counts['val']} classes")
    print(f"  Test:  {counts['test']} classes")

    # Update config
    import yaml
    config_path = os.path.join(PROJECT_DIR, "configs", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["num_classes"] = total_classes
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Updated config with num_classes={total_classes}")


if __name__ == "__main__":
    organize_dataset()
