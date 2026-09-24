"""Phase 8.4 - Verify Streamlit's shared prediction integration."""

import json
import sys
from pathlib import Path

from PIL import Image

from model_predictor import CLASS_NAMES
from prediction_pipeline import predict_document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset" / "test"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "streamlit_integration_verification.json"
STREAMLIT_ENTRY_POINT = PROJECT_ROOT / "app.py"


def main():
    sample_paths = sorted(TEST_ROOT.rglob("*.png"))
    checks = {
        "streamlit_entry_point_exists": STREAMLIT_ENTRY_POINT.is_file(),
        "predictor_imported": False,
        "pipeline_imported": False,
        "sample_image_loaded": False,
        "prediction_result_complete": False,
        "predicted_class_valid": False,
        "confidence_valid": False,
        "probabilities_valid": False,
    }
    result = None
    sample_path = None
    error = None

    try:
        checks["predictor_imported"] = True
        checks["pipeline_imported"] = callable(predict_document)
        if not sample_paths:
            raise FileNotFoundError("No PNG test image was found")
        sample_path = sample_paths[0]
        with Image.open(sample_path) as image:
            image.load()
            checks["sample_image_loaded"] = True
            result = predict_document(image)
        required_fields = {
            "predicted_class",
            "predicted_class_index",
            "confidence",
            "confidence_percentage",
            "probabilities",
        }
        checks["prediction_result_complete"] = required_fields <= set(result)
        checks["predicted_class_valid"] = result["predicted_class"] in CLASS_NAMES
        checks["confidence_valid"] = 0.0 <= result["confidence"] <= 1.0
        probabilities = result["probabilities"]
        values = [probabilities[name] for name in CLASS_NAMES]
        checks["probabilities_valid"] = (
            set(probabilities) == set(CLASS_NAMES)
            and all(0.0 <= value <= 1.0 for value in values)
            and abs(sum(values) - 1.0) < 1e-5
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        error = str(exc)

    passed = all(checks.values()) and error is None
    report = {
        "streamlit_entry_point": STREAMLIT_ENTRY_POINT.relative_to(PROJECT_ROOT).as_posix(),
        "predictor_import_status": checks["predictor_imported"],
        "pipeline_import_status": checks["pipeline_imported"],
        "sample_image": sample_path.relative_to(PROJECT_ROOT).as_posix() if sample_path else None,
        "prediction": result,
        "integration_checks": checks,
        "error": error,
        "verification_status": "PASS" if passed else "FAIL",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print("\n" + "=" * 72)
    print("SMARTDOCS AI - PHASE 8.4 STREAMLIT INTEGRATION")
    print("=" * 72)
    print(f"Streamlit entry point   : {report['streamlit_entry_point']}")
    print(f"Predictor import status : {'PASS' if checks['predictor_imported'] else 'FAIL'}")
    print(f"Pipeline import status  : {'PASS' if checks['pipeline_imported'] else 'FAIL'}")
    print(f"Sample image            : {report['sample_image']}")
    if result:
        print(f"Prediction              : {result['predicted_class']}")
        print(f"Confidence              : {result['confidence_percentage']:.2f}%")
        print("Probability distribution:")
        for class_name in CLASS_NAMES:
            print(f"  {class_name:<10}: {result['probabilities'][class_name]:.6f}")
    if error:
        print(f"Error                   : {error}")
    print("\nIntegration checks:")
    for check_name, check_passed in checks.items():
        print(f"  {check_name:<30}: {'PASS' if check_passed else 'FAIL'}")
    print(f"\nVerification report     : {REPORT_PATH}")
    print(f"STREAMLIT INTEGRATION: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
