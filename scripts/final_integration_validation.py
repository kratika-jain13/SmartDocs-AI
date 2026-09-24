"""Phase 8.8 - Final read-only integration validation."""

import json
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import prediction_pipeline
from scripts.model_predictor import CLASS_NAMES, SmartDocsPredictor

TEST_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset" / "test"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "final_integration_validation.json"
EXPECTED_CLASS_TO_INDEX = {
    "forms": 0,
    "invoices": 1,
    "other": 2,
    "resumes": 3,
}
IMPORTANT_FILES = [
    "app.py",
    "models/mobilenet_v3_small_classifier_best.pth",
    "scripts/model_predictor.py",
    "scripts/prediction_pipeline.py",
]
PHASE_8_REPORTS = [
    "data/metadata/final_model_verification.json",
    "data/metadata/model_predictor_verification.json",
    "data/metadata/prediction_pipeline_verification.json",
    "data/metadata/streamlit_integration_verification.json",
    "data/metadata/prediction_result_ui_verification.json",
    "data/metadata/end_to_end_test_report.json",
]


def path_exists(relative_path):
    return (PROJECT_ROOT / relative_path).is_file()


def main():
    checks = {
        "file_checks": {path: path_exists(path) for path in IMPORTANT_FILES},
        "import_checks": {
            "prediction_pipeline_imported": False,
            "predict_document_imported": False,
            "model_predictor_imported": False,
            "smartdocs_predictor_imported": False,
        },
        "model_checks": {
            "predictor_initialized": False,
            "model_in_evaluation_mode": False,
            "four_output_classes": False,
        },
        "class_mapping_checks": {
            "mapping_matches_expected": False,
        },
        "prediction_checks": {
            "sample_image_loaded": False,
            "predicted_class_valid": False,
            "confidence_valid": False,
            "confidence_percentage_valid": False,
            "all_probabilities_present": False,
            "probabilities_valid": False,
            "probabilities_sum_to_one": False,
        },
        "streamlit_checks": {"entry_point_exists": path_exists("app.py")},
        "report_checks": {path: path_exists(path) for path in PHASE_8_REPORTS},
    }
    errors = []
    prediction = None
    sample_image = None
    predictor = None

    try:
        from scripts.prediction_pipeline import predict_document

        checks["import_checks"]["prediction_pipeline_imported"] = True
        checks["import_checks"]["predict_document_imported"] = callable(predict_document)

        checks["import_checks"]["model_predictor_imported"] = True
        checks["import_checks"]["smartdocs_predictor_imported"] = SmartDocsPredictor is not None

        predictor = SmartDocsPredictor()
        checks["model_checks"]["predictor_initialized"] = True
        checks["model_checks"]["model_in_evaluation_mode"] = not predictor.model.training
        checks["model_checks"]["four_output_classes"] = (
            predictor.model.classifier[-1].out_features == 4
        )
        checks["class_mapping_checks"]["mapping_matches_expected"] = (
            predictor.class_to_index == EXPECTED_CLASS_TO_INDEX
        )

        sample_paths = sorted(TEST_ROOT.rglob("*.png"))
        if not sample_paths:
            raise FileNotFoundError("No PNG test image was found")
        sample_image = sample_paths[0]
        with Image.open(sample_image) as image:
            image.load()
            checks["prediction_checks"]["sample_image_loaded"] = True
            prediction = prediction_pipeline.predict_document(image)

        checks["prediction_checks"]["predicted_class_valid"] = (
            prediction.get("predicted_class") in CLASS_NAMES
        )
        checks["prediction_checks"]["confidence_valid"] = (
            0.0 <= prediction.get("confidence", -1) <= 1.0
        )
        checks["prediction_checks"]["confidence_percentage_valid"] = (
            0.0 <= prediction.get("confidence_percentage", -1) <= 100.0
        )
        probabilities = prediction.get("probabilities", {})
        checks["prediction_checks"]["all_probabilities_present"] = (
            set(probabilities) == set(CLASS_NAMES)
        )
        values = [probabilities.get(class_name, -1) for class_name in CLASS_NAMES]
        checks["prediction_checks"]["probabilities_valid"] = all(
            0.0 <= value <= 1.0 for value in values
        )
        checks["prediction_checks"]["probabilities_sum_to_one"] = (
            abs(sum(values) - 1.0) < 1e-5
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(str(exc))

    all_checks = [
        result
        for group in checks.values()
        for result in group.values()
    ]
    passed = not errors and all(all_checks)
    report = {
        "file_checks": checks["file_checks"],
        "import_checks": checks["import_checks"],
        "model_checks": checks["model_checks"],
        "class_mapping_checks": checks["class_mapping_checks"],
        "prediction_checks": checks["prediction_checks"],
        "streamlit_checks": checks["streamlit_checks"],
        "report_checks": checks["report_checks"],
        "sample_image": sample_image.relative_to(PROJECT_ROOT).as_posix() if sample_image else None,
        "prediction": prediction,
        "errors": errors,
        "final_status": "PASS" if passed else "FAIL",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print("\n" + "=" * 76)
    print("SMARTDOCS AI - PHASE 8.8 FINAL INTEGRATION VALIDATION")
    print("=" * 76)
    for title, group_name in (
        ("File checks", "file_checks"),
        ("Import checks", "import_checks"),
        ("Model checks", "model_checks"),
        ("Class mapping checks", "class_mapping_checks"),
        ("Prediction checks", "prediction_checks"),
        ("Streamlit checks", "streamlit_checks"),
        ("Report checks", "report_checks"),
    ):
        print(f"\n{title}")
        for check_name, result in checks[group_name].items():
            print(f"  {check_name:<42}: {'PASS' if result else 'FAIL'}")
    if prediction:
        print(f"\nSample image            : {report['sample_image']}")
        print(f"Predicted class         : {prediction['predicted_class']}")
        print(f"Confidence              : {prediction['confidence_percentage']:.2f}%")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
    print(f"\nValidation report      : {REPORT_PATH}")
    print(f"FINAL INTEGRATION VALIDATION: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
