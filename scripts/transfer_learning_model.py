"""Phase 7.5 - Configure a MobileNetV3-Small transfer-learning model."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
TRAIN_ROOT = DATASET_ROOT / "train"
VALIDATION_ROOT = DATASET_ROOT / "validation"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
REPORT_PATH = METADATA_DIR / "transfer_learning_setup.json"
BATCH_SIZE = 32
NUM_CLASSES = 4
EXPECTED_CLASS_TO_INDEX = {
    "forms": 0,
    "invoices": 1,
    "other": 2,
    "resumes": 3,
}


def fail(message):
    print(f"[ERROR] {message}")
    print("TRANSFER LEARNING SETUP: FAIL")
    return 1


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def main():
    print("\n" + "=" * 72)
    print("SMARTDOCS AI - PHASE 7.5 TRANSFER LEARNING MODEL SETUP")
    print("=" * 72)
    print(f"Training dataset         : {TRAIN_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"Validation dataset       : {VALIDATION_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"Batch size               : {BATCH_SIZE}")
    print("Data augmentation        : disabled")
    print("Test set                 : not loaded or evaluated")

    try:
        import torch
        import torch.nn as nn
        import torchvision
        from torchvision import datasets
        from torchvision.models import MobileNet_V3_Small_Weights
        from torchvision.models import mobilenet_v3_small
    except ImportError as exc:
        return fail(f"PyTorch/torchvision import failed: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version          : {torch.__version__}")
    print(f"torchvision version      : {torchvision.__version__}")
    print(f"Selected device         : {device}")

    if not TRAIN_ROOT.is_dir():
        return fail(f"Training directory does not exist: {TRAIN_ROOT}")
    if not VALIDATION_ROOT.is_dir():
        return fail(f"Validation directory does not exist: {VALIDATION_ROOT}")

    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
    except Exception as exc:
        return fail(
            "Could not load MobileNetV3-Small ImageNet weights. "
            f"Check the installed torchvision version and network/cache access: {exc}"
        )

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    classifier_input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(classifier_input_features, NUM_CLASSES)
    model = model.to(device)

    transform = weights.transforms()
    try:
        train_dataset = datasets.ImageFolder(str(TRAIN_ROOT), transform=transform)
        validation_dataset = datasets.ImageFolder(
            str(VALIDATION_ROOT), transform=transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return fail(f"Could not load ImageFolder datasets or DataLoaders: {exc}")

    if train_dataset.class_to_idx != EXPECTED_CLASS_TO_INDEX:
        return fail(
            "ImageFolder class mapping differs from the required mapping: "
            f"{train_dataset.class_to_idx}"
        )
    if validation_dataset.class_to_idx != train_dataset.class_to_idx:
        return fail("Validation ImageFolder mapping differs from training mapping")
    if len(train_dataset.classes) != NUM_CLASSES:
        return fail(f"Expected {NUM_CLASSES} classes, found {len(train_dataset.classes)}")

    total_parameters, trainable_parameters = parameter_counts(model)
    print(f"\nModel name              : torchvision.models.mobilenet_v3_small")
    print(f"Number of classes      : {NUM_CLASSES}")
    print(f"Class names             : {train_dataset.classes}")
    print(f"Class-to-index mapping  : {train_dataset.class_to_idx}")
    print(f"Total parameters        : {total_parameters:,}")
    print(f"Trainable parameters    : {trainable_parameters:,}")
    print("Pretrained weights      : ImageNet MobileNet_V3_Small_Weights.DEFAULT")
    print(f"Train images            : {len(train_dataset)}")
    print(f"Validation images       : {len(validation_dataset)}")

    try:
        batch_images, batch_labels = next(iter(train_loader))
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(batch_images)
            sample_loss = nn.CrossEntropyLoss()(outputs, batch_labels)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        return fail(f"Transfer-learning forward pass failed: {exc}")

    expected_output_shape = (batch_images.shape[0], NUM_CLASSES)
    print("\nForward-pass verification:")
    print(f"  Batch shape            : {tuple(batch_images.shape)}")
    print(f"  Output shape           : {tuple(outputs.shape)}")
    print(f"  Sample CrossEntropyLoss: {sample_loss.item():.6f}")
    if tuple(outputs.shape) != expected_output_shape:
        return fail(
            f"Expected output shape {expected_output_shape}, got {tuple(outputs.shape)}"
        )
    if not torch.isfinite(sample_loss):
        return fail("Sample CrossEntropyLoss is not finite")

    report = {
        "architecture": "torchvision.models.mobilenet_v3_small",
        "pretrained_status": {
            "requested": True,
            "weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "loaded": True,
        },
        "class_names": train_dataset.classes,
        "class_to_index": train_dataset.class_to_idx,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_feature_layers": True,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "output_shape": list(outputs.shape),
        "train_image_count": len(train_dataset),
        "validation_image_count": len(validation_dataset),
        "test_set_evaluated": False,
        "data_augmentation": False,
    }
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(f"\nSetup report saved      : {REPORT_PATH}")
    print("Training                : not performed")
    print("TRANSFER LEARNING SETUP: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
