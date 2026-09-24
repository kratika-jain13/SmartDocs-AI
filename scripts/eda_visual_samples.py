"""Phase 6.4 - Visual sample analysis for the final dataset.

The script selects readable images only and creates a read-only visualization.
It never changes, resizes, overwrites, or converts any dataset image.
"""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
REPORT_FILE = METADATA_DIR / "eda_visual_samples.json"
VISUALIZATION_FILE = METADATA_DIR / "eda_visual_samples.png"

RANDOM_SEED = 42
SAMPLES_PER_CLASS_PER_SPLIT = 2
SPLITS = ("train", "validation", "test")
CLASSES = ("invoices", "forms", "resumes", "other")
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def readable_images(directory, split, class_name, unreadable_images):
    """Return readable image paths while recording files that cannot be opened."""
    candidates = []
    try:
        paths = sorted(directory.iterdir())
    except OSError as exc:
        unreadable_images.append(
            {
                "split": split,
                "class": class_name,
                "path": directory.relative_to(PROJECT_ROOT).as_posix(),
                "error": str(exc),
            }
        )
        return candidates

    for path in paths:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            unreadable_images.append(
                {
                    "split": split,
                    "class": class_name,
                    "filename": path.name,
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "error": str(exc),
                }
            )
            continue
        candidates.append(path)
    return candidates


def select_samples():
    """Select a deterministic, small sample for every class and split."""
    generator = random.Random(RANDOM_SEED)
    selected_samples = {class_name: [] for class_name in CLASSES}
    unreadable_images = []
    missing_directories = []

    for split in SPLITS:
        for class_name in CLASSES:
            directory = DATASET_ROOT / split / class_name
            if not directory.is_dir():
                missing_directories.append(
                    directory.relative_to(PROJECT_ROOT).as_posix()
                )
                continue

            candidates = readable_images(
                directory, split, class_name, unreadable_images
            )
            sample_count = min(SAMPLES_PER_CLASS_PER_SPLIT, len(candidates))
            for path in generator.sample(candidates, sample_count):
                selected_samples[class_name].append(
                    {
                        "class": class_name,
                        "split": split,
                        "filename": path.name,
                        "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    }
                )

    return {
        "selected_samples": selected_samples,
        "unreadable_images": unreadable_images,
        "missing_directories": missing_directories,
    }


def create_visualization(selected_samples, plot_failures):
    """Create one row per class with selected images and descriptive captions."""
    maximum_columns = max(
        (len(samples) for samples in selected_samples.values()), default=0
    )
    column_count = max(1, maximum_columns)
    figure, axes = plt.subplots(
        len(CLASSES),
        column_count,
        figsize=(3.4 * column_count, 3.2 * len(CLASSES)),
        squeeze=False,
    )

    for row, class_name in enumerate(CLASSES):
        samples = selected_samples[class_name]
        for column in range(column_count):
            axis = axes[row][column]
            axis.axis("off")
            if column >= len(samples):
                continue
            sample = samples[column]
            path = PROJECT_ROOT / sample["path"]
            try:
                with Image.open(path) as image:
                    axis.imshow(image.copy())
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                plot_failures.append({**sample, "error": str(exc)})
                continue
            axis.set_title(
                f"{sample['class']}\n{sample['split']}\n{sample['filename']}",
                fontsize=8,
            )

    figure.suptitle("SmartDocs AI Visual Dataset Samples", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(VISUALIZATION_FILE, dpi=150)
    plt.close(figure)


def build_report(selection, plot_failures):
    """Build the JSON report with reproducibility and selection details."""
    selected = selection["selected_samples"]
    return {
        "dataset_root": DATASET_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "random_seed": RANDOM_SEED,
        "requested_samples_per_class_per_split": SAMPLES_PER_CLASS_PER_SPLIT,
        "number_of_samples_per_class": {
            class_name: len(selected[class_name]) for class_name in CLASSES
        },
        "classes_analyzed": list(CLASSES),
        "splits_analyzed": list(SPLITS),
        "selected_image_filenames": {
            class_name: [sample["filename"] for sample in selected[class_name]]
            for class_name in CLASSES
        },
        "selected_image_paths": {
            class_name: [sample["path"] for sample in selected[class_name]]
            for class_name in CLASSES
        },
        "selected_samples": selected,
        "checks": {
            "unreadable_image_count": len(selection["unreadable_images"]),
            "unreadable_images": selection["unreadable_images"],
            "missing_directory_count": len(selection["missing_directories"]),
            "missing_directories": selection["missing_directories"],
            "visualization_failure_count": len(plot_failures),
            "visualization_failures": plot_failures,
        },
    }


def print_report(report):
    """Print all selected samples and safety checks."""
    print("\n" + "=" * 68)
    print("SMARTDOCS AI - VISUAL SAMPLE ANALYSIS")
    print("=" * 68)
    print(f"Dataset root             : {report['dataset_root']}")
    print(f"Random seed              : {report['random_seed']}")
    print(
        "Samples per class/split  : "
        f"{report['requested_samples_per_class_per_split']}"
    )
    print("\nSelected images:")
    for class_name in report["classes_analyzed"]:
        print(f"  {class_name} ({report['number_of_samples_per_class'][class_name]} total):")
        for sample in report["selected_samples"][class_name]:
            print(f"    [{sample['split']}] {sample['path']}")

    checks = report["checks"]
    print("\nChecks:")
    print(f"  Unreadable images      : {checks['unreadable_image_count']}")
    print(f"  Missing directories    : {checks['missing_directory_count']}")
    print(f"  Visualization failures : {checks['visualization_failure_count']}")
    print(f"\nVisualization written to : {VISUALIZATION_FILE}")
    print(f"Report written to        : {REPORT_FILE}")


def main():
    """Select samples, overwrite the repeatable report, and create the grid."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_ROOT}")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    selection = select_samples()
    plot_failures = []
    create_visualization(selection["selected_samples"], plot_failures)
    report = build_report(selection, plot_failures)
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print_report(report)


if __name__ == "__main__":
    main()