"""Phase 8.2 - Verify the reusable SmartDocs predictor."""

import json
import sys
from pathlib import Path

from PIL import Image

from model_predictor import CLASS_NAMES, CLASS_TO_INDEX, SmartDocsPredictor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset" / "test"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "model_predictor_verification.json"


def main():
    sample_paths = sorted(TEST_ROOT.rglob("*.png"))
    if not sample_paths:
        print("[ERROR] No PNG test image was found")
        print("MODEL PREDICTOR VERIFICATION: FAIL")
        return 1

    sample_path = sample_paths[0]
    try:
        predictor = SmartDocsPredictor()
        with Image.open(sample_path) as image:
            result = predictor.predict(image)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] Predictor verification failed: {exc}")
        print("MODEL PREDICTOR VERIFICATION: FAIL")
        return 1

    probabilities = result["probabilities"]
    probability_values = list(probabilities.values())
    checks = {
        "model_loaded": predictor.model is not None,
        "predicted_class_valid": result["predicted_class"] in CLASS_NAMES,
        "predicted_index_valid": result["predicted_class_index"] in range(len(CLASS_NAMES)),
        "confidence_valid": 0.0 <= result["confidence"] <= 1.0,
        "probabilities_valid": (
            len(probability_values) == len(CLASS_NAMES)
            and all(0.0 <= value <= 1.0 for value in probability_values)
        ),
        "probabilities_sum_to_one": abs(sum(probability_values) - 1.0) < 1e-5,
        "class_mapping_valid": predictor.class_to_index == CLASS_TO_INDEX,
        "model_in_eval_mode": not predictor.model.training,
    }
    verification_passed = all(checks.values())
    report = {
        "model_loaded": checks["model_loaded"],
        "device": str(predictor.device),
        "class_names": predictor.get_class_names(),
        "class_to_index": predictor.class_to_index,
        "sample_image": sample_path.relative_to(PROJECT_ROOT).as_posix(),
        "prediction": result,
        "verification_checks": checks,
        "verification_status": "PASS" if verification_passed else "FAIL",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print("\n" + "=" * 72)
    print("SMARTDOCS AI - PHASE 8.2 MODEL PREDICTOR VERIFICATION")
    print("=" * 72)
    print(f"Model loaded            : {checks['model_loaded']}")
    print(f"Device                  : {predictor.device}")
    print(f"Class mapping           : {predictor.class_to_index}")
    print(f"Sample image            : {report['sample_image']}")
    print(f"Prediction              : {result['predicted_class']}")
    print(f"Predicted class index   : {result['predicted_class_index']}")
    print(f"Confidence              : {result['confidence']:.6f}")
    print("Probability distribution:")
    for class_name, probability in probabilities.items():
        print(f"  {class_name:<10}: {probability:.6f}")
    print("\nVerification checks:")
    for check_name, passed in checks.items():
        print(f"  {check_name:<28}: {'PASS' if passed else 'FAIL'}")
    print(f"\nVerification report     : {REPORT_PATH}")
    print(
        "MODEL PREDICTOR VERIFICATION: "
        f"{'PASS' if verification_passed else 'FAIL'}"
    )
    return 0 if verification_passed else 1


if __name__ == "__main__":
    sys.exit(main())
