"""Phase 8.1 - Verify the selected final model with one test image."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset" / "test"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "mobilenet_v3_small_classifier_best.pth"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "final_model_verification.json"
NUM_CLASSES = 4
EXPECTED_CLASS_TO_INDEX = {
    "forms": 0,
    "invoices": 1,
    "other": 2,
    "resumes": 3,
}
CLASS_NAMES = ["forms", "invoices", "other", "resumes"]
ARCHITECTURE = "torchvision.models.mobilenet_v3_small"
PRETRAINED_WEIGHTS = "MobileNet_V3_Small_Weights.DEFAULT"


def fail(message):
    print(f"[ERROR] {message}")
    print("FINAL MODEL VERIFICATION: FAIL")
    return 1


def create_model(torch, nn):
    """Recreate the exact Phase 7.5/7.6 classifier architecture."""
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
            f"Checkpoint architecture is {checkpoint.get('architecture')!r}; "
            f"expected {ARCHITECTURE!r}"
        )
    if checkpoint.get("class_to_index") not in (None, EXPECTED_CLASS_TO_INDEX):
        raise ValueError("Checkpoint class mapping differs from the required mapping")


def main():
    print("\n" + "=" * 76)
    print("SMARTDOCS AI - PHASE 8.1 FINAL MODEL VERIFICATION")
    print("=" * 76)
    print(f"Checkpoint path         : {CHECKPOINT_PATH}")
    print(f"Model architecture      : {ARCHITECTURE}")
    print(f"Test dataset            : {TEST_ROOT.relative_to(PROJECT_ROOT)}")

    try:
        import torch
        import torch.nn as nn
        import torchvision
        from torchvision import datasets
        from torchvision.models import MobileNet_V3_Small_Weights
    except ImportError as exc:
        return fail(f"PyTorch/torchvision import failed: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not TEST_ROOT.is_dir():
        return fail(f"Test dataset does not exist: {TEST_ROOT}")
    if not CHECKPOINT_PATH.is_file():
        return fail(f"Final model checkpoint does not exist: {CHECKPOINT_PATH}")

    try:
        # Use the same ImageNet normalization and 224x224 preprocessing as evaluation.
        weights = MobileNet_V3_Small_Weights.DEFAULT
        test_dataset = datasets.ImageFolder(
            root=str(TEST_ROOT),
            transform=weights.transforms(),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return fail(f"Could not load the test ImageFolder dataset: {exc}")

    if not test_dataset.samples:
        return fail("The test dataset contains no valid images")
    if test_dataset.class_to_idx != EXPECTED_CLASS_TO_INDEX:
        return fail(
            "Test ImageFolder mapping differs from the required mapping: "
            f"{test_dataset.class_to_idx}"
        )
    if test_dataset.classes != CLASS_NAMES:
        return fail(f"Unexpected test class order: {test_dataset.classes}")

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )
        validate_checkpoint(checkpoint)
        model = create_model(torch, nn).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except (OSError, RuntimeError, ValueError) as exc:
        return fail(f"Could not load and prepare the final model: {exc}")

    classifier_output_count = model.classifier[-1].out_features
    if classifier_output_count != NUM_CLASSES:
        return fail(
            f"Final classifier has {classifier_output_count} outputs; "
            f"expected {NUM_CLASSES}"
        )

    sample_path, expected_index = test_dataset.samples[0]
    try:
        image_tensor, expected_label = test_dataset[0]
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0).to(device))
            probabilities = torch.softmax(output, dim=1)
            predicted_index = int(probabilities.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_index].item())
    except (OSError, RuntimeError, ValueError) as exc:
        return fail(f"Single-image inference failed: {exc}")

    output_shape = list(output.shape)
    probability_values = probabilities[0].detach().cpu().tolist()
    probabilities_valid = (
        len(probability_values) == NUM_CLASSES
        and all(0.0 <= value <= 1.0 for value in probability_values)
        and abs(sum(probability_values) - 1.0) < 1e-5
    )
    output_shape_valid = output_shape == [1, NUM_CLASSES]
    predicted_class_valid = predicted_index in range(NUM_CLASSES)
    confidence_valid = 0.0 <= confidence <= 1.0
    checks = {
        "checkpoint_exists": CHECKPOINT_PATH.is_file(),
        "model_in_eval_mode": not model.training,
        "classifier_outputs_four_classes": classifier_output_count == NUM_CLASSES,
        "class_mapping_valid": test_dataset.class_to_idx == EXPECTED_CLASS_TO_INDEX,
        "probabilities_valid": probabilities_valid,
        "predicted_class_valid": predicted_class_valid,
        "confidence_valid": confidence_valid,
        "output_shape_valid": output_shape_valid,
    }
    verification_passed = all(checks.values())
    predicted_class = CLASS_NAMES[predicted_index] if predicted_class_valid else None

    report = {
        "checkpoint_path": CHECKPOINT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "model_architecture": ARCHITECTURE,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "device": str(device),
        "class_names": CLASS_NAMES,
        "class_to_index": EXPECTED_CLASS_TO_INDEX,
        "sample_image": Path(sample_path).relative_to(PROJECT_ROOT).as_posix(),
        "expected_class_index": int(expected_index),
        "expected_class": CLASS_NAMES[expected_label],
        "predicted_class_index": predicted_index,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "probabilities": {
            class_name: round(float(probability_values[index]), 6)
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "model_output_shape": output_shape,
        "verification_checks": checks,
        "verification_status": "PASS" if verification_passed else "FAIL",
    }
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
    except OSError as exc:
        return fail(f"Could not save verification report: {exc}")

    print(f"Device                  : {device}")
    print(f"Class mapping           : {EXPECTED_CLASS_TO_INDEX}")
    print(f"Model output shape      : {output_shape}")
    print(f"Sample image            : {report['sample_image']}")
    print(f"Predicted class         : {predicted_class}")
    print(f"Predicted class index   : {predicted_index}")
    print(f"Confidence              : {confidence:.6f}")
    print("\nVerification checks:")
    for check_name, passed in checks.items():
        print(f"  {check_name:<32}: {'PASS' if passed else 'FAIL'}")
    print(f"\nVerification report     : {REPORT_PATH}")
    print(
        "FINAL MODEL VERIFICATION: "
        f"{'PASS' if verification_passed else 'FAIL'}"
    )
    return 0 if verification_passed else 1


if __name__ == "__main__":
    sys.exit(main())
