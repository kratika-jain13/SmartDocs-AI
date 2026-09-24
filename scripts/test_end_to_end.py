"""Phase 8.7 - End-to-end prediction workflow test."""

import json
import sys
from pathlib import Path

from PIL import Image

# Allow direct execution from the project root while preserving the package
# import used by the Streamlit application.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_predictor import CLASS_NAMES
from prediction_pipeline import predict_document


TEST_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset" / "test"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "end_to_end_test_report.json"
TEST_CLASSES = ["forms", "invoices", "other", "resumes"]


def validate_result(result):
    required_fields = {
        "predicted_class",
        "predicted_class_index",
        "confidence",
        "confidence_percentage",
        "probabilities",
    }
    if not required_fields.issubset(result):
        return False, "Missing prediction result fields"

    probabilities = result["probabilities"]
    values = [probabilities.get(class_name) for class_name in CLASS_NAMES]
    if set(probabilities) != set(CLASS_NAMES):
        return False, "Probability distribution does not contain all classes"
    if any(value is None or not 0.0 <= value <= 1.0 for value in values):
        return False, "Probability value is outside [0, 1]"
    if abs(sum(values) - 1.0) >= 1e-5:
        return False, "Probabilities do not sum to approximately 1"
    if result["predicted_class"] not in CLASS_NAMES:
        return False, "Predicted class is invalid"
    if result["predicted_class_index"] not in range(len(CLASS_NAMES)):
        return False, "Predicted class index is invalid"
    if not 0.0 <= result["confidence"] <= 1.0:
        return False, "Confidence is outside [0, 1]"
    if not 0.0 <= result["confidence_percentage"] <= 100.0:
        return False, "Confidence percentage is outside [0, 100]"
    return True, "All prediction result checks passed"


def main():
    sample_results = []
    successful_tests = 0
    failed_tests = 0

    for class_name in TEST_CLASSES:
        class_directory = TEST_ROOT / class_name
        sample_paths = sorted(class_directory.glob("*.png"))
        result_record = {
            "class_sample": class_name,
            "input_image": None,
            "predicted_class": None,
            "confidence": None,
            "probability_validation": False,
            "status": "FAIL",
            "error": None,
        }
        try:
            if not sample_paths:
                raise FileNotFoundError(f"No PNG sample found in {class_directory}")
            sample_path = sample_paths[0]
            result_record["input_image"] = sample_path.relative_to(PROJECT_ROOT).as_posix()
            with Image.open(sample_path) as image:
                image.load()
                result = predict_document(image)
            valid, message = validate_result(result)
            result_record.update(
                {
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"],
                    "confidence_percentage": result["confidence_percentage"],
                    "probability_validation": valid,
                    "status": "PASS" if valid else "FAIL",
                    "validation_message": message,
                }
            )
            if valid:
                successful_tests += 1
            else:
                failed_tests += 1
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            result_record["error"] = str(exc)
            result_record["validation_message"] = "Prediction test raised an error"
            failed_tests += 1
        sample_results.append(result_record)

    invalid_input_passed = False
    invalid_input_error = None
    try:
        predict_document(None)
        invalid_input_error = "None input was unexpectedly accepted"
    except (TypeError, ValueError) as exc:
        invalid_input_passed = True
        invalid_input_error = str(exc)

    total_tests = len(TEST_CLASSES)
    overall_passed = failed_tests == 0 and invalid_input_passed
    report = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": failed_tests,
        "sample_results": sample_results,
        "invalid_input_test": {
            "input": None,
            "passed": invalid_input_passed,
            "handled_error": invalid_input_error,
        },
        "overall_status": "PASS" if overall_passed else "FAIL",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print("\n" + "=" * 88)
    print("SMARTDOCS AI - PHASE 8.7 END-TO-END TEST")
    print("=" * 88)
    print("Class/sample                 Input image                                  Prediction  Confidence  Probability validation  Status")
    for result_record in sample_results:
        print(
            f"{result_record['class_sample']:<28} "
            f"{str(result_record['input_image']):<44} "
            f"{str(result_record['predicted_class']):<11} "
            f"{str(result_record['confidence']):<11} "
            f"{str(result_record['probability_validation']):<22} "
            f"{result_record['status']}"
        )
    print("\nFinal summary:")
    print(f"  Total tests             : {total_tests}")
    print(f"  Successful tests       : {successful_tests}")
    print(f"  Failed tests            : {failed_tests}")
    print(f"  Invalid-input test     : {'PASS' if invalid_input_passed else 'FAIL'}")
    print(f"  Invalid-input handling : {invalid_input_error}")
    print(f"  Report                 : {REPORT_PATH}")
    print(f"\nEND-TO-END TEST: {'PASS' if overall_passed else 'FAIL'}")
    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
