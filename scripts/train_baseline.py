"""Phase 7.3 - Train the SmartDocs AI baseline CNN.

Only the train and validation splits are used here. The test split is not
loaded or evaluated until the final evaluation phase.
"""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "baseline_cnn_best.pth"
HISTORY_PATH = PROJECT_ROOT / "data" / "metadata" / "baseline_training_history.json"

SPLITS = ("train", "validation")
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


def fail(message):
    """Print a clear training error and return a failed result."""
    print(f"[ERROR] {message}")
    return False


def create_model(torch, nn):
    """Create the same small CNN architecture used in Phase 7.2."""
    class BaselineCNN(nn.Module):
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

    return BaselineCNN(NUM_CLASSES)


def run_epoch(model, loader, criterion, device, torch, training):
    """Run one training or validation epoch and return loss and accuracy."""
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer = run_epoch.optimizer
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                run_epoch.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    """Train for the configured epochs and save the best validation model."""
    print("\n" + "=" * 72)
    print("SMARTDOCS AI - PHASE 7.3 BASELINE MODEL TRAINING")
    print("=" * 72)
    print(f"Dataset root             : {DATASET_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"Epochs                   : {EPOCHS}")
    print(f"Batch size               : {BATCH_SIZE}")
    print(f"Learning rate            : {LEARNING_RATE}")
    print("Data augmentation        : disabled")
    print("Test set                 : not loaded or evaluated")

    try:
        import torch
        import torch.nn as nn
        import torchvision
        from torchvision import datasets, transforms
    except ImportError as exc:
        fail(f"PyTorch/torchvision import failed: {exc}")
        print("BASELINE TRAINING: FAIL")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Keep the small CPU baseline responsive on constrained machines.
        torch.set_num_threads(1)
    print(f"PyTorch version          : {torch.__version__}")
    print(f"torchvision version      : {torchvision.__version__}")
    print(f"Selected device          : {device}")

    if not DATASET_ROOT.is_dir():
        fail(f"Dataset root does not exist: {DATASET_ROOT}")
        print("BASELINE TRAINING: FAIL")
        return 1

    datasets_by_split = {}
    loaders_by_split = {}
    transform = transforms.ToTensor()
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
        fail(f"Could not load training datasets: {exc}")
        print("BASELINE TRAINING: FAIL")
        return 1

    train_dataset = datasets_by_split.get("train")
    validation_dataset = datasets_by_split.get("validation")
    if train_dataset is None or validation_dataset is None:
        print("BASELINE TRAINING: FAIL")
        return 1

    print(f"\nClass names              : {train_dataset.classes}")
    print(f"Class-to-index mapping  : {train_dataset.class_to_idx}")
    if train_dataset.class_to_idx != EXPECTED_CLASS_TO_INDEX:
        setup_ok = fail("ImageFolder mapping differs from Phase 7.1")
    if validation_dataset.class_to_idx != train_dataset.class_to_idx:
        setup_ok = fail("Validation class mapping differs from training")

    print(f"Train images             : {len(train_dataset)}")
    print(f"Validation images        : {len(validation_dataset)}")

    model = create_model(torch, nn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Keep the optimizer accessible to the small epoch helper without changing
    # the model or data-loading API.
    run_epoch.optimizer = optimizer
    history = {
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "best_validation_accuracy": None,
        "best_epoch": None,
        "class_names": train_dataset.classes,
        "class_to_index": train_dataset.class_to_idx,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "device": str(device),
    }
    best_validation_accuracy = -1.0
    best_epoch = None

    print("\nTraining history:")
    print("Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy")
    print("------|------------|----------------|----------|-------------")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders_by_split["train"], criterion, device, torch, training=True
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            loaders_by_split["validation"],
            criterion,
            device,
            torch,
            training=False,
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
                },
                MODEL_PATH,
            )

    history["best_validation_accuracy"] = round(best_validation_accuracy, 6)
    history["best_epoch"] = best_epoch
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
        file.write("\n")

    if not MODEL_PATH.is_file():
        setup_ok = fail(f"Best model was not saved: {MODEL_PATH}")
    if best_epoch is None:
        setup_ok = fail("No best validation epoch was recorded")

    print("\nFinal results:")
    print(f"  Best epoch              : {best_epoch}")
    print(f"  Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"  Model path              : {MODEL_PATH}")
    print(f"  Training history path   : {HISTORY_PATH}")

    if setup_ok:
        print("\nBASELINE TRAINING: PASS")
        return 0
    print("\nBASELINE TRAINING: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())