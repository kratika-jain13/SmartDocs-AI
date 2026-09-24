"""Phase 7.8 - Evaluate the trained transfer-learning model on the test set."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset" / "test"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "mobilenet_v3_small_classifier_best.pth"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
CONFUSION_MATRIX_PATH = METADATA_DIR / "final_confusion_matrix.png"
REPORT_PATH = METADATA_DIR / "final_test_evaluation.json"
BATCH_SIZE = 32
NUM_CLASSES = 4
EXPECTED_TEST_SAMPLES = 300
EXPECTED_CLASS_TO_INDEX = {
    "forms": 0,
    "invoices": 1,
    "other": 2,
    "resumes": 3,
}
CLASS_NAMES = ["forms", "invoices", "other", "resumes"]
ARCHITECTURE = "torchvision.models.mobilenet_v3_small"


def fail(message):
    print(f"[ERROR] {message}")
    print("FINAL TEST EVALUATION: FAIL")
    return 1


def create_model(torch, nn):
    """Recreate the Phase 7.5/7.6 MobileNetV3-Small classifier."""
    import torchvision

    model = torchvision.models.mobilenet_v3_small(weights=None)
    classifier_input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(classifier_input_features, NUM_CLASSES)
    return model


def validate_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict")
    if checkpoint.get("architecture") not in (None, ARCHITECTURE):
        raise ValueError(
            f"Checkpoint architecture is {checkpoint.get('architecture')!r}, "
            f"expected {ARCHITECTURE!r}"
        )
    if checkpoint.get("class_to_index") not in (None, EXPECTED_CLASS_TO_INDEX):
        raise ValueError("Checkpoint class mapping differs from the required mapping")


def save_confusion_matrix(matrix, class_names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(f"Matplotlib import failed: {exc}") from exc

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Final Test Confusion Matrix",
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(figure)


def main():
    print("\n" + "=" * 76)
    print("SMARTDOCS AI - PHASE 7.8 FINAL TEST EVALUATION")
    print("=" * 76)
    print(f"Model name              : {ARCHITECTURE}")
    print(f"Checkpoint path         : {CHECKPOINT_PATH}")
    print(f"Test dataset            : {TEST_ROOT.relative_to(PROJECT_ROOT)}")

    try:
        import torch
        import torch.nn as nn
        import torchvision
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        from torchvision import datasets
        from torchvision.models import MobileNet_V3_Small_Weights
    except ImportError as exc:
        return fail(f"Required evaluation dependency is unavailable: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not TEST_ROOT.is_dir():
        return fail(f"Test directory does not exist: {TEST_ROOT}")
    if not CHECKPOINT_PATH.is_file():
        return fail(f"Model checkpoint does not exist: {CHECKPOINT_PATH}")

    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        test_dataset = datasets.ImageFolder(str(TEST_ROOT), transform=weights.transforms())
    except (OSError, RuntimeError, ValueError) as exc:
        return fail(f"Could not load test ImageFolder dataset: {exc}")

    if len(test_dataset) != EXPECTED_TEST_SAMPLES:
        return fail(
            f"Expected exactly {EXPECTED_TEST_SAMPLES} test images, found {len(test_dataset)}"
        )
    if test_dataset.class_to_idx != EXPECTED_CLASS_TO_INDEX:
        return fail(
            "Test ImageFolder class mapping differs from the required mapping: "
            f"{test_dataset.class_to_idx}"
        )
    if test_dataset.classes != CLASS_NAMES:
        return fail(f"Unexpected test class order: {test_dataset.classes}")

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        validate_checkpoint(checkpoint)
        model = create_model(torch, nn).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return fail(f"Could not load model checkpoint or prepare inference: {exc}")

    all_labels = []
    all_predictions = []
    try:
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(device))
                predictions = outputs.argmax(dim=1).cpu().tolist()
                all_predictions.extend(predictions)
                all_labels.extend(labels.tolist())
    except (OSError, RuntimeError, ValueError) as exc:
        return fail(f"Test inference failed: {exc}")

    labels = list(range(NUM_CLASSES))
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions, labels=labels, average="weighted", zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions, labels=labels, average="weighted", zero_division=0
    )
    f1 = f1_score(
        all_labels, all_predictions, labels=labels, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(all_labels, all_predictions, labels=labels)
    class_report = classification_report(
        all_labels,
        all_predictions,
        labels=labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    correct = int(sum(actual == predicted for actual, predicted in zip(all_labels, all_predictions)))
    incorrect = len(all_labels) - correct

    try:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(matrix, CLASS_NAMES)
        report = {
            "model_name": ARCHITECTURE,
            "checkpoint_path": CHECKPOINT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "device": str(device),
            "class_names": CLASS_NAMES,
            "class_to_index": EXPECTED_CLASS_TO_INDEX,
            "total_test_samples": len(all_labels),
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "test_accuracy": round(float(accuracy), 6),
            "weighted_precision": round(float(precision), 6),
            "weighted_recall": round(float(recall), 6),
            "weighted_f1_score": round(float(f1), 6),
            "per_class_metrics": {
                class_name: {
                    "precision": round(float(class_report[class_name]["precision"]), 6),
                    "recall": round(float(class_report[class_name]["recall"]), 6),
                    "f1_score": round(float(class_report[class_name]["f1-score"]), 6),
                    "support": int(class_report[class_name]["support"]),
                }
                for class_name in CLASS_NAMES
            },
            "confusion_matrix": matrix.tolist(),
            "confusion_matrix_path": CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "evaluation_status": "PASS",
        }
        with REPORT_PATH.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
    except (OSError, RuntimeError, ValueError) as exc:
        return fail(f"Could not save evaluation outputs: {exc}")

    print(f"Device                  : {device}")
    print(f"Test samples            : {len(all_labels)}")
    print(f"Correct predictions     : {correct}")
    print(f"Incorrect predictions   : {incorrect}")
    print(f"Test accuracy           : {accuracy:.6f}")
    print(f"Weighted precision      : {precision:.6f}")
    print(f"Weighted recall         : {recall:.6f}")
    print(f"Weighted F1-score       : {f1:.6f}")
    print("\nPer-class metrics:")
    print("Class       Precision   Recall   F1-score   Support")
    for class_name in CLASS_NAMES:
        metrics = report["per_class_metrics"][class_name]
        print(
            f"{class_name:<11} {metrics['precision']:>9.6f} "
            f"{metrics['recall']:>8.6f} {metrics['f1_score']:>10.6f} "
            f"{metrics['support']:>9}"
        )
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print("Class order: forms, invoices, other, resumes")
    for row in matrix.tolist():
        print(f"  {row}")
    print(f"\nConfusion matrix saved   : {CONFUSION_MATRIX_PATH}")
    print(f"Evaluation report saved  : {REPORT_PATH}")
    print("FINAL TEST EVALUATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
