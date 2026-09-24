"""Phase 7.1 - Verify the ML training environment and dataset loading.

This script only checks the environment and reads the finalized dataset. It
does not train a model, augment images, or modify dataset files.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
SPLITS = ("train", "validation", "test")
EXPECTED_CLASSES = {"invoices", "forms", "resumes", "other"}
EXPECTED_COUNTS = {"train": 1400, "validation": 300, "test": 300}
BATCH_SIZE = 32
EXPECTED_IMAGE_SHAPE = (3, 224, 224)


def fail(message):
    """Print a clear failure message and return a failed setup result."""
    print(f"[ERROR] {message}")
    return False


def main():
    """Run all environment, dataset, and DataLoader checks."""
    print("\n" + "=" * 68)
    print("SMARTDOCS AI - PHASE 7.1 ML TRAINING SETUP")
    print("=" * 68)
    print(f"Dataset root             : {DATASET_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"Batch size               : {BATCH_SIZE}")
    print("Data augmentation        : disabled")

    try:
        import torch
    except ImportError as exc:
        fail(f"PyTorch is not installed or could not be imported: {exc}")
        print("TRAINING SETUP: FAIL")
        return 1

    try:
        import torchvision
        from torchvision import datasets, transforms
    except ImportError as exc:
        fail(f"torchvision is not installed or could not be imported: {exc}")
        print(f"PyTorch version          : {torch.__version__}")
        print("TRAINING SETUP: FAIL")
        return 1

    print(f"PyTorch installed        : yes ({torch.__version__})")
    print(f"torchvision installed    : yes ({torchvision.__version__})")
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"CUDA/GPU available       : {cuda_available}")
    print(f"Selected device          : {device}")

    if not DATASET_ROOT.is_dir():
        fail(f"Dataset root does not exist: {DATASET_ROOT}")
        print("TRAINING SETUP: FAIL")
        return 1

    # ToTensor is required to batch images as tensors; it performs no
    # augmentation and does not write anything to the source dataset.
    transform = transforms.ToTensor()
    datasets_by_split = {}
    loaders_by_split = {}
    setup_ok = True

    try:
        for split in SPLITS:
            split_path = DATASET_ROOT / split
            if not split_path.is_dir():
                setup_ok = fail(f"Missing {split} split directory: {split_path}")
                continue
            dataset = datasets.ImageFolder(root=str(split_path), transform=transform)
            datasets_by_split[split] = dataset
            loaders_by_split[split] = torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=(split == "train"),
                num_workers=0,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        fail(f"Could not load dataset or create DataLoaders: {exc}")
        print("TRAINING SETUP: FAIL")
        return 1

    train_dataset = datasets_by_split.get("train")
    if train_dataset is None:
        print("TRAINING SETUP: FAIL")
        return 1

    print(f"\nDetected class names    : {train_dataset.classes}")
    print(f"Class-to-index mapping  : {train_dataset.class_to_idx}")
    if set(train_dataset.classes) != EXPECTED_CLASSES:
        setup_ok = fail(
            f"Detected classes do not match expected classes: {EXPECTED_CLASSES}"
        )

    for split in SPLITS:
        dataset = datasets_by_split.get(split)
        count = len(dataset) if dataset is not None else 0
        expected = EXPECTED_COUNTS[split]
        print(f"{split.capitalize():<24}: {count} (expected {expected})")
        if count != expected:
            setup_ok = fail(
                f"{split} image count is {count}; expected {expected}"
            )

    for split in SPLITS:
        dataset = datasets_by_split.get(split)
        if dataset is None or dataset.classes != train_dataset.classes:
            setup_ok = fail(f"Class mapping differs for {split}")

    try:
        batch_images, batch_labels = next(iter(loaders_by_split["train"]))
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        setup_ok = fail(f"Could not load a training batch: {exc}")
    else:
        expected_batch_shape = (min(BATCH_SIZE, EXPECTED_COUNTS["train"]),) + EXPECTED_IMAGE_SHAPE
        print("\nTraining batch:")
        print(f"  Image tensor shape     : {tuple(batch_images.shape)}")
        print(f"  Label tensor shape     : {tuple(batch_labels.shape)}")
        print(f"  First few labels       : {batch_labels[:5].tolist()}")
        print(f"  Tensor data type       : {batch_images.dtype}")
        if tuple(batch_images.shape) != expected_batch_shape:
            setup_ok = fail(
                f"Unexpected batch image shape: {tuple(batch_images.shape)}; "
                f"expected {expected_batch_shape}"
            )
        if tuple(batch_labels.shape) != expected_batch_shape[:1]:
            setup_ok = fail(
                f"Unexpected batch label shape: {tuple(batch_labels.shape)}"
            )
        if batch_images.dtype != torch.float32:
            setup_ok = fail(f"Unexpected tensor data type: {batch_images.dtype}")

    if setup_ok:
        print("\nTRAINING SETUP: PASS")
        return 0
    print("\nTRAINING SETUP: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())