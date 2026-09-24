"""Reusable SmartDocs AI predictor for the final image classifier."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "mobilenet_v3_small_classifier_best.pth"
)
CLASS_TO_INDEX = {
    "forms": 0,
    "invoices": 1,
    "other": 2,
    "resumes": 3,
}
CLASS_NAMES = ["forms", "invoices", "other", "resumes"]
NUM_CLASSES = 4
ARCHITECTURE = "torchvision.models.mobilenet_v3_small"
PRETRAINED_WEIGHTS = "MobileNet_V3_Small_Weights.DEFAULT"


class SmartDocsPredictor:
    """Load the final SmartDocs model and predict document classes."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT_PATH):
        import torch
        import torch.nn as nn
        import torchvision
        from torchvision.models import MobileNet_V3_Small_Weights

        self.torch = torch
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_to_index = CLASS_TO_INDEX.copy()
        self.class_names = CLASS_NAMES.copy()
        self.weights = MobileNet_V3_Small_Weights.DEFAULT
        self.transform = self.weights.transforms()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Final model checkpoint does not exist: {self.checkpoint_path}"
            )

        # Recreate the classifier architecture without downloading weights;
        # the trained state is loaded from the project checkpoint below.
        self.model = torchvision.models.mobilenet_v3_small(weights=None)
        classifier_input_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(
            classifier_input_features, NUM_CLASSES
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self._validate_checkpoint(checkpoint)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _validate_checkpoint(self, checkpoint):
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary")
        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model_state_dict")
        if checkpoint.get("architecture") not in (None, ARCHITECTURE):
            raise ValueError("Checkpoint architecture does not match the predictor")
        if checkpoint.get("class_to_index") not in (None, CLASS_TO_INDEX):
            raise ValueError("Checkpoint class mapping does not match the predictor")

    def predict(self, image):
        """Return the predicted class and softmax probability distribution."""
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("predict(image) expects a PIL.Image.Image instance")

        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = self.torch.softmax(logits, dim=1)[0]

        predicted_index = int(probabilities.argmax().item())
        probability_values = probabilities.detach().cpu().tolist()
        return {
            "predicted_class": self.class_names[predicted_index],
            "predicted_class_index": predicted_index,
            "confidence": float(probability_values[predicted_index]),
            "probabilities": {
                class_name: float(probability_values[index])
                for index, class_name in enumerate(self.class_names)
            },
        }

    def get_class_names(self):
        """Return class names in model output-index order."""
        return self.class_names.copy()
