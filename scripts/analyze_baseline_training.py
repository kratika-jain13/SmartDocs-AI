"""Phase 7.4 - Analyze baseline training history without retraining."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
HISTORY_PATH = METADATA_DIR / "baseline_training_history.json"
LOSS_PLOT_PATH = METADATA_DIR / "baseline_loss_curve.png"
ACCURACY_PLOT_PATH = METADATA_DIR / "baseline_accuracy_curve.png"
ANALYSIS_PATH = METADATA_DIR / "baseline_training_analysis.json"
REQUIRED_SERIES = (
    "epochs",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
)


def load_history():
    if not HISTORY_PATH.is_file():
        raise FileNotFoundError(f"Training history not found: {HISTORY_PATH}")
    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as file:
            history = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read training history: {exc}") from exc

    if not isinstance(history, dict):
        raise ValueError("Training history must be a JSON object.")
    missing = [key for key in REQUIRED_SERIES if key not in history]
    if missing:
        raise ValueError(f"Training history is missing: {', '.join(missing)}")

    lengths = {key: len(history[key]) for key in REQUIRED_SERIES}
    if len(set(lengths.values())) != 1 or not lengths["epochs"]:
        raise ValueError(f"Training series have inconsistent lengths: {lengths}")
    return history


def rounded(value):
    return round(float(value), 6)


def calculate_statistics(history):
    epochs = history["epochs"]
    train_accuracy = history["train_accuracy"]
    validation_accuracy = history["validation_accuracy"]
    validation_loss = history["validation_loss"]
    best_index = max(range(len(validation_accuracy)), key=validation_accuracy.__getitem__)

    return {
        "initial_training_accuracy": rounded(train_accuracy[0]),
        "final_training_accuracy": rounded(train_accuracy[-1]),
        "best_validation_accuracy": rounded(validation_accuracy[best_index]),
        "best_validation_accuracy_epoch": epochs[best_index],
        "initial_validation_loss": rounded(validation_loss[0]),
        "final_validation_loss": rounded(validation_loss[-1]),
    }


def create_plots(history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(f"Matplotlib import failed: {exc}") from exc

    epochs = history["epochs"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
    axis.plot(epochs, history["validation_loss"], marker="o", label="Validation Loss")
    axis.set_title("Training and Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_xticks(epochs)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["train_accuracy"], marker="o", label="Training Accuracy")
    axis.plot(epochs, history["validation_accuracy"], marker="o", label="Validation Accuracy")
    axis.set_title("Training and Validation Accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(epochs)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(ACCURACY_PLOT_PATH, dpi=150)
    plt.close(figure)


def factual_observations(history, statistics):
    observations = [
        (
            f"Training accuracy changed from {statistics['initial_training_accuracy']:.6f} "
            f"at epoch {history['epochs'][0]} to "
            f"{statistics['final_training_accuracy']:.6f} at epoch {history['epochs'][-1]}."
        ),
        (
            f"Validation accuracy reached its highest recorded value of "
            f"{statistics['best_validation_accuracy']:.6f} at epoch "
            f"{statistics['best_validation_accuracy_epoch']}."
        ),
        (
            f"Validation loss changed from {statistics['initial_validation_loss']:.6f} "
            f"at epoch {history['epochs'][0]} to "
            f"{statistics['final_validation_loss']:.6f} at epoch {history['epochs'][-1]}."
        ),
    ]
    if history["train_loss"][-1] < history["train_loss"][0]:
        observations.append(
            f"Training loss decreased from {history['train_loss'][0]:.6f} "
            f"to {history['train_loss'][-1]:.6f} over the recorded epochs."
        )
    else:
        observations.append(
            f"Training loss changed from {history['train_loss'][0]:.6f} "
            f"to {history['train_loss'][-1]:.6f} over the recorded epochs."
        )
    return observations


def print_summary(history, statistics, observations):
    print("\n" + "=" * 72)
    print("SMARTDOCS AI - PHASE 7.4 BASELINE TRAINING ANALYSIS")
    print("=" * 72)
    print(f"History file             : {HISTORY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Epochs analyzed          : {len(history['epochs'])}")
    print(f"Initial training accuracy: {statistics['initial_training_accuracy']:.6f}")
    print(f"Final training accuracy  : {statistics['final_training_accuracy']:.6f}")
    print(f"Best validation accuracy : {statistics['best_validation_accuracy']:.6f}")
    print(f"Best validation epoch    : {statistics['best_validation_accuracy_epoch']}")
    print(f"Initial validation loss  : {statistics['initial_validation_loss']:.6f}")
    print(f"Final validation loss    : {statistics['final_validation_loss']:.6f}")
    print("\nFactual observations:")
    for observation in observations:
        print(f"  - {observation}")
    print("\nGenerated files:")
    print(f"  Loss plot              : {LOSS_PLOT_PATH}")
    print(f"  Accuracy plot          : {ACCURACY_PLOT_PATH}")
    print(f"  Analysis report        : {ANALYSIS_PATH}")


def main():
    try:
        history = load_history()
        statistics = calculate_statistics(history)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        create_plots(history)
        observations = factual_observations(history, statistics)
        report = {
            "training_summary": {
                "epochs": history["epochs"],
                "class_names": history.get("class_names", []),
                "class_to_index": history.get("class_to_index", {}),
                "learning_rate": history.get("learning_rate"),
                "batch_size": history.get("batch_size"),
                "device": history.get("device"),
            },
            "calculated_statistics": statistics,
            "best_epoch": statistics["best_validation_accuracy_epoch"],
            "best_validation_accuracy": statistics["best_validation_accuracy"],
            "metric_series": {
                "train_loss": history["train_loss"],
                "train_accuracy": history["train_accuracy"],
                "validation_loss": history["validation_loss"],
                "validation_accuracy": history["validation_accuracy"],
            },
            "plot_paths": {
                "loss": LOSS_PLOT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "accuracy": ACCURACY_PLOT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            },
            "factual_observations": observations,
        }
        with ANALYSIS_PATH.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_summary(history, statistics, observations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
