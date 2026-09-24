"""Phase 8.3 - Reusable document prediction pipeline."""

from PIL import Image

from scripts.model_predictor import CLASS_NAMES, SmartDocsPredictor

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = SmartDocsPredictor()
    return _predictor


def predict_document(image):
    """Predict a document class from a PIL image.

    Raises:
        TypeError: If ``image`` is not a PIL Image.
        ValueError: If the image cannot be converted or produces invalid output.
    """
    if image is None:
        raise ValueError("image must not be None")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance")

    try:
        rgb_image = image.convert("RGB")
        result = _get_predictor().predict(rgb_image)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not process image: {exc}") from exc

    if result["predicted_class"] not in CLASS_NAMES:
        raise ValueError("Predictor returned an unknown class")
    if not 0 <= result["confidence"] <= 1:
        raise ValueError("Predictor returned an invalid confidence")

    probabilities = result["probabilities"]
    if set(probabilities) != set(CLASS_NAMES):
        raise ValueError("Predictor did not return probabilities for all classes")
    probability_values = [probabilities[class_name] for class_name in CLASS_NAMES]
    if (
        any(not 0 <= value <= 1 for value in probability_values)
        or abs(sum(probability_values) - 1.0) >= 1e-5
    ):
        raise ValueError("Predictor returned an invalid probability distribution")

    return {
        "predicted_class": result["predicted_class"],
        "predicted_class_index": result["predicted_class_index"],
        "confidence": result["confidence"],
        "confidence_percentage": result["confidence"] * 100.0,
        "probabilities": {
            class_name: probabilities[class_name] for class_name in CLASS_NAMES
        },
    }
