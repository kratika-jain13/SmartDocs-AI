"""Phase 8.3 - Verify the reusable prediction pipeline."""

import json
import sys
from pathlib import Path

from PIL import Image

from model_predictor import CLASS_NAMES, CLASS_TO_INDEX
from prediction_pipeline import predict_document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset" / "test"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "prediction_pipeline_verification.json"


def main():
    sample_paths = sorted(TEST_ROOT.rglob("*.png"))
    if not sample_paths:
        print("[ERROR] No PNG test image was found")
        print("PREDICTION PIPELINE: FAIL")
        return 1

    sample_path = sample_paths[0]
    try:
        with Image.open(sample_path) as image:
            result = predict_document(image)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] Prediction pipeline failed: {exc}")
        print("PREDICTION PIPELINE: FAIL")
        return 1

    probabilities = result["probabilities"]
    probability_values = [probabilities[class_name] for class_name in CLASS_NAMES]
    checks = {
        "predicted_class_valid": result["predicted_class"] in CLASS_NAMES,
        "predicted_index_valid": result["predicted_class_index"] in range(len(CLASS_NAMES)),
        "confidence_valid": 0.0 <= result["confidence"] <= 1.0,
        "confidence_percentage_valid": 0.0 <= result["confidence_percentage"] <= 100.0,
        "all_classes_present": set(probabilities) == set(CLASS_NAMES),
        "probabilities_valid": all(0.0 <= value <= 1.0 for value in probability_values),
        "probabilities_sum_to_one": abs(sum(probability_values) - 1.0) < 1e-5,
        "class_mapping_valid": CLASS_TO_INDEX == {
            "forms": 0,
            "invoices": 1,
            "other": 2,
            "resumes": 3,
        },
    }
    verification_passed = all(checks.values())
    report = {
        "pipeline_status": "PASS" if verification_passed else "FAIL",
        "sample_image": sample_path.relative_to(PROJECT_ROOT).as_posix(),
        "class_names": CLASS_NAMES,
        "class_to_index": CLASS_TO_INDEX,
        "prediction": result,
        "validation_checks": checks,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print("\n" + "=" * 72)
    print("SMARTDOCS AI - PHASE 8.3 PREDICTION PIPELINE")
    print("=" * 72)
    print(f"Pipeline status         : {report['pipeline_status']}")
    print(f"Sample image            : {report['sample_image']}")
    print(f"Predicted class         : {result['predicted_class']}")
    print(f"Predicted class index   : {result['predicted_class_index']}")
    print(f"Confidence              : {result['confidence']:.6f}")
    print(f"Confidence percentage   : {result['confidence_percentage']:.2f}%")
    print("Probability distribution:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name:<10}: {probabilities[class_name]:.6f}")
    print("\nValidation checks:")
    for check_name, passed in checks.items():
        print(f"  {check_name:<28}: {'PASS' if passed else 'FAIL'}")
    print(f"\nVerification report     : {REPORT_PATH}")
    print(f"PREDICTION PIPELINE: {'PASS' if verification_passed else 'FAIL'}")
    return 0 if verification_passed else 1


if __name__ == "__main__":
    sys.exit(main())
