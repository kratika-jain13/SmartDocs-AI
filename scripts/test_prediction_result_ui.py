"""Phase 8.6 - Verify the prediction result UI data contract."""

import json
import sys
from pathlib import Path

from PIL import Image

# Allow this script to run directly from the project root while preserving the
# package import used by the Streamlit entry point.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prediction_pipeline import predict_document


TEST_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset" / "test"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "prediction_result_ui_verification.json"
CLASS_NAMES = ["forms", "invoices", "other", "resumes"]


def main():
    sample_paths = sorted(TEST_ROOT.rglob("*.png"))
    checks = {
        "prediction_result_present": False,
        "predicted_class_present": False,
        "confidence_present": False,
        "confidence_percentage_present": False,
        "all_four_probabilities_present": False,
        "probability_values_valid": False,
        "probabilities_sum_to_one": False,
    }
    result = None
    sample_path = sample_paths[0] if sample_paths else None
    error = None

    try:
        if sample_path is None:
            raise FileNotFoundError("No PNG test image was found")
        with Image.open(sample_path) as image:
            result = predict_document(image)
        checks["prediction_result_present"] = isinstance(result, dict)
        checks["predicted_class_present"] = bool(result.get("predicted_class"))
        checks["confidence_present"] = "confidence" in result
        checks["confidence_percentage_present"] = "confidence_percentage" in result
        probabilities = result.get("probabilities", {})
        checks["all_four_probabilities_present"] = set(probabilities) == set(CLASS_NAMES)
        values = [probabilities[name] for name in CLASS_NAMES]
        checks["probability_values_valid"] = (
            len(values) == len(CLASS_NAMES)
            and all(0.0 <= value <= 1.0 for value in values)
        )
        checks["probabilities_sum_to_one"] = abs(sum(values) - 1.0) < 1e-5
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        error = str(exc)

    passed = all(checks.values()) and error is None
    report = {
        "prediction_result_status": "PASS" if passed else "FAIL",
        "sample_image": sample_path.relative_to(PROJECT_ROOT).as_posix() if sample_path else None,
        "prediction": result,
        "validation_checks": checks,
        "error": error,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print("\n" + "=" * 72)
    print("SMARTDOCS AI - PHASE 8.6 PREDICTION RESULT UI")
    print("=" * 72)
    print(f"Prediction result status : {report['prediction_result_status']}")
    print(f"Sample image             : {report['sample_image']}")
    if result:
        print(f"Predicted class          : {result['predicted_class']}")
        print(f"Confidence               : {result['confidence_percentage']:.2f}%")
    if error:
        print(f"Error                    : {error}")
    print("\nProbability validation:")
    for check_name, check_passed in checks.items():
        print(f"  {check_name:<30}: {'PASS' if check_passed else 'FAIL'}")
    print(f"\nVerification report      : {REPORT_PATH}")
    print(f"PREDICTION RESULT UI: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
