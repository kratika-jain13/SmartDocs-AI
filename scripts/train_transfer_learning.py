"""Phase 7.6 - Train the frozen-feature MobileNetV3-Small classifier."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
TRAIN_ROOT = DATASET_ROOT / "train"
VALIDATION_ROOT = DATASET_ROOT / "validation"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "mobilenet_v3_small_classifier_best.pth"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
HISTORY_PATH = METADATA_DIR / "transfer_learning_training_history.json"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
NUM_CLASSES = 4
EXPECTED_CLASS_TO_INDEX = {
    "forms": 0,
    "invoices": 1,
    "other": 2,
    "resumes": 3,
}
ARCHITECTURE = "torchvision.models.mobilenet_v3_small"
PRETRAINED_WEIGHTS = "MobileNet_V3_Small_Weights.DEFAULT"


def fail(message):
    print(f"[ERROR] {message}")
    print("TRANSFER LEARNING TRAINING: FAIL")
    return 1


def create_model(torch, nn, weights):
    """Create Phase 7.5's model and leave only its final layer trainable."""
    model = __import__("torchvision").models.mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    classifier_input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(classifier_input_features, NUM_CLASSES)
    return model


def run_epoch(model, loader, criterion, optimizer, device, training):
    """Run one train or validation epoch and return mean loss and accuracy."""
    if training:
        model.train()
        # Keep frozen feature parameters and BatchNorm buffers unchanged.
        model.features.eval()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with __import__("torch").set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    print("\n" + "=" * 72)
    print("SMARTDOCS AI - PHASE 7.6 TRANSFER LEARNING TRAINING")
    print("=" * 72)
    print(f"Training dataset         : {TRAIN_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"Validation dataset       : {VALIDATION_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"Architecture             : {ARCHITECTURE}")
    print(f"Epochs                   : {EPOCHS}")
    print(f"Batch size               : {BATCH_SIZE}")
    print(f"Learning rate            : {LEARNING_RATE}")
    print("Pretrained weights       : ImageNet")
    print("Frozen feature extractor : yes")
    print("Data augmentation        : disabled")
    print("Test set                 : not loaded or evaluated")

    try:
        import torch
        import torch.nn as nn
        import torchvision
        from torchvision import datasets
        from torchvision.models import MobileNet_V3_Small_Weights
    except ImportError as exc:
        return fail(f"PyTorch/torchvision import failed: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    print(f"PyTorch version          : {torch.__version__}")
    print(f"torchvision version      : {torchvision.__version__}")
    print(f"Selected device         : {device}")

    if not TRAIN_ROOT.is_dir():
        return fail(f"Training directory does not exist: {TRAIN_ROOT}")
    if not VALIDATION_ROOT.is_dir():
        return fail(f"Validation directory does not exist: {VALIDATION_ROOT}")

    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = create_model(torch, nn, weights).to(device)
    except Exception as exc:
        return fail(
            "Could not load MobileNetV3-Small ImageNet weights. "
            f"Check torchvision version and network/cache access: {exc}"
        )

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

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters or any(
        parameter.requires_grad for parameter in model.features.parameters()
    ):
        return fail("Feature extractor is not fully frozen")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_parameters, lr=LEARNING_RATE)
    history = {
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "best_epoch": None,
        "best_validation_accuracy": None,
        "class_names": train_dataset.classes,
        "class_to_index": train_dataset.class_to_idx,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "architecture": ARCHITECTURE,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "frozen_feature_status": True,
    }
    best_validation_accuracy = -1.0
    best_epoch = None

    print(f"\nClass names              : {train_dataset.classes}")
    print(f"Class-to-index mapping   : {train_dataset.class_to_idx}")
    print(f"Train images             : {len(train_dataset)}")
    print(f"Validation images        : {len(validation_dataset)}")
    print(f"Trainable parameters     : {sum(p.numel() for p in trainable_parameters):,}")
    print("\nTraining history:")
    print("Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy")
    print("------|------------|----------------|----------|-------------")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, optimizer, device, training=False
        )
        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["train_accuracy"].append(round(train_accuracy, 6))
        history["validation_loss"].append(round(validation_loss, 6))
        history["validation_accuracy"].append(round(validation_accuracy, 6))
        print(
            f"{epoch:5d} | {train_loss:10.4f} | {train_accuracy:14.4f} | "
            f"{validation_loss:8.4f} | {validation_accuracy:11.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": train_dataset.classes,
                    "class_to_index": train_dataset.class_to_idx,
                    "validation_accuracy": validation_accuracy,
                    "architecture": ARCHITECTURE,
                    "pretrained_weights": PRETRAINED_WEIGHTS,
                    "frozen_feature_status": True,
                },
                MODEL_PATH,
            )

    history["best_epoch"] = best_epoch
    history["best_validation_accuracy"] = round(best_validation_accuracy, 6)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
        file.write("\n")

    if best_epoch is None or not MODEL_PATH.is_file():
        return fail("Best transfer-learning model was not saved")

    print("\nFinal results:")
    print(f"  Best epoch              : {best_epoch}")
    print(f"  Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"  Model path              : {MODEL_PATH}")
    print(f"  Training history path   : {HISTORY_PATH}")
    print("\nTest set                 : not evaluated")
    print("TRANSFER LEARNING TRAINING: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
