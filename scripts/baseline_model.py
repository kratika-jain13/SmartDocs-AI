"""Phase 7.2 - Simple CNN baseline setup.

This script verifies a baseline model with one forward pass and a sample loss.
It does not train, augment, or modify the dataset.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
SPLITS = ("train", "validation", "test")
EXPECTED_CLASS_TO_INDEX = {
    "forms": 0,
    "invoices": 1,
    "other": 2,
    "resumes": 3,
}
BATCH_SIZE = 32
NUM_CLASSES = 4
EXPECTED_IMAGE_SHAPE = (3, 224, 224)


def fail(message):
    """Print a clear setup error and return a failed check."""
    print(f"[ERROR] {message}")
    return False


def main():
    """Load data, construct the CNN, and validate one forward pass."""
    print("\n" + "=" * 68)
    print("SMARTDOCS AI - PHASE 7.2 BASELINE MODEL")
    print("=" * 68)
    print(f"Dataset root             : {DATASET_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"Batch size               : {BATCH_SIZE}")
    print("Data augmentation        : disabled")
    print("Pretrained model         : not used")

    try:
        import torch
        import torch.nn as nn
        import torchvision
        from torchvision import datasets, transforms
    except ImportError as exc:
        fail(f"PyTorch/torchvision import failed: {exc}")
        print("BASELINE MODEL SETUP: FAIL")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version          : {torch.__version__}")
    print(f"torchvision version      : {torchvision.__version__}")
    print(f"Selected device          : {device}")

    class BaselineCNN(nn.Module):
        """Small CNN suitable for a CPU-only baseline forward pass."""

        def __init__(self, number_of_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, number_of_classes),
            )

        def forward(self, images):
            return self.classifier(self.features(images))

    if not DATASET_ROOT.is_dir():
        fail(f"Dataset root does not exist: {DATASET_ROOT}")
        print("BASELINE MODEL SETUP: FAIL")
        return 1

    transform = transforms.ToTensor()
    datasets_by_split = {}
    loaders_by_split = {}
    setup_ok = True

    try:
        for split in SPLITS:
            split_path = DATASET_ROOT / split
            if not split_path.is_dir():
                setup_ok = fail(f"Missing {split} directory: {split_path}")
                continue
            dataset = datasets.ImageFolder(str(split_path), transform=transform)
            datasets_by_split[split] = dataset
            loaders_by_split[split] = torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=(split == "train"),
                num_workers=0,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        fail(f"Could not load ImageFolder datasets or DataLoaders: {exc}")
        print("BASELINE MODEL SETUP: FAIL")
        return 1

    train_dataset = datasets_by_split.get("train")
    if train_dataset is None:
        print("BASELINE MODEL SETUP: FAIL")
        return 1

    print(f"\nNumber of classes        : {len(train_dataset.classes)}")
    print(f"Class names              : {train_dataset.classes}")
    print(f"Class-to-index mapping  : {train_dataset.class_to_idx}")
    if train_dataset.class_to_idx != EXPECTED_CLASS_TO_INDEX:
        setup_ok = fail(
            "ImageFolder mapping differs from the required Phase 7.1 mapping"
        )
    if len(train_dataset.classes) != NUM_CLASSES:
        setup_ok = fail(f"Expected {NUM_CLASSES} classes")

    for split in SPLITS:
        dataset = datasets_by_split.get(split)
        if dataset is None or dataset.class_to_idx != train_dataset.class_to_idx:
            setup_ok = fail(f"Class mapping differs for {split}")

    print("\nDataset image counts:")
    for split in SPLITS:
        dataset = datasets_by_split.get(split)
        count = len(dataset) if dataset is not None else 0
        print(f"  {split.capitalize():<20}: {count}")

    try:
        model = BaselineCNN(NUM_CLASSES).to(device)
        model.eval()
        print("\nModel architecture:")
        print(model)

        batch_images, batch_labels = next(iter(loaders_by_split["train"]))
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            outputs = model(batch_images)
            loss = nn.CrossEntropyLoss()(outputs, batch_labels)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        setup_ok = fail(f"Baseline forward pass failed: {exc}")
    else:
        print("\nForward-pass verification:")
        print(f"  Batch shape             : {tuple(batch_images.shape)}")
        print(f"  Model output shape      : {tuple(outputs.shape)}")
        print(f"  Sample loss             : {loss.item():.6f}")
        expected_batch_shape = (min(BATCH_SIZE, len(train_dataset)),) + EXPECTED_IMAGE_SHAPE
        expected_output_shape = (expected_batch_shape[0], NUM_CLASSES)
        if tuple(batch_images.shape) != expected_batch_shape:
            setup_ok = fail(
                f"Expected batch shape {expected_batch_shape}, "
                f"got {tuple(batch_images.shape)}"
            )
        if tuple(outputs.shape) != expected_output_shape:
            setup_ok = fail(
                f"Expected output shape {expected_output_shape}, "
                f"got {tuple(outputs.shape)}"
            )
        if not torch.isfinite(loss):
            setup_ok = fail("Sample loss is not finite")

    if setup_ok:
        print("\nBASELINE MODEL SETUP: PASS")
        return 0
    print("\nBASELINE MODEL SETUP: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())