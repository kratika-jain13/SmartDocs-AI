"""Phase 6.2 - Class distribution analysis for the final dataset.

The analysis is read-only: it counts files, validates readable images, writes
an analysis report, and creates a chart without changing dataset contents.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
REPORT_FILE = METADATA_DIR / "class_distribution.json"
CHART_FILE = METADATA_DIR / "class_distribution.png"

SPLITS = ("train", "validation", "test")
CLASSES = ("invoices", "forms", "resumes", "other")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def scan_dataset():
    """Count image files and record files that cannot be read safely."""
    class_counts = {
        split: {class_name: 0 for class_name in CLASSES} for split in SPLITS
    }
    unreadable_files = []
    missing_directories = []

    for split in SPLITS:
        for class_name in CLASSES:
            directory = DATASET_ROOT / split / class_name
            if not directory.is_dir():
                missing_directories.append(
                    directory.relative_to(PROJECT_ROOT).as_posix()
                )
                continue

            try:
                paths = sorted(directory.iterdir())
            except OSError as exc:
                missing_directories.append(
                    {
                        "path": directory.relative_to(PROJECT_ROOT).as_posix(),
                        "error": str(exc),
                    }
                )
                continue

            for path in paths:
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Count image files even when unreadable, then expose the issue
                # so the distribution remains transparent and reproducible.
                class_counts[split][class_name] += 1
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    unreadable_files.append(
                        {
                            "split": split,
                            "class": class_name,
                            "filename": path.name,
                            "path": path.relative_to(PROJECT_ROOT).as_posix(),
                            "error": str(exc),
                        }
                    )

    split_counts = {
        split: sum(class_counts[split].values()) for split in SPLITS
    }
    complete_counts = {
        class_name: sum(class_counts[split][class_name] for split in SPLITS)
        for class_name in CLASSES
    }
    complete_counts_total = sum(complete_counts.values())
    return {
        "class_counts": class_counts,
        "split_counts": split_counts,
        "complete_counts": complete_counts,
        "complete_counts_total": complete_counts_total,
        "unreadable_files": unreadable_files,
        "missing_directories": missing_directories,
    }


def percentages(counts, total):
    """Return class percentages, using zero for an empty split."""
    return {
        class_name: round((counts[class_name] / total) * 100, 2) if total else 0.0
        for class_name in CLASSES
    }


def build_report(scan):
    """Build the required JSON report from the read-only scan."""
    class_percentages = {
        split: percentages(scan["class_counts"][split], scan["split_counts"][split])
        for split in SPLITS
    }
    class_percentages["complete_dataset"] = percentages(
        scan["complete_counts"], scan["complete_counts_total"]
    )

    balanced_within_splits = all(
        len(set(scan["class_counts"][split].values())) == 1 for split in SPLITS
    )
    balanced_complete_dataset = len(set(scan["complete_counts"].values())) == 1

    return {
        "dataset_root": DATASET_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "total_images": scan["complete_counts_total"],
        "split_counts": scan["split_counts"],
        "class_counts": {
            **scan["class_counts"],
            "complete_dataset": scan["complete_counts"],
        },
        "class_percentages": class_percentages,
        "balance_status": {
            "status": (
                "BALANCED"
                if balanced_within_splits and balanced_complete_dataset
                else "NOT_BALANCED"
            ),
            "is_balanced": balanced_within_splits and balanced_complete_dataset,
            "balanced_within_each_split": balanced_within_splits,
            "balanced_in_complete_dataset": balanced_complete_dataset,
        },
        "file_quality": {
            "unreadable_file_count": len(scan["unreadable_files"]),
            "unreadable_files": scan["unreadable_files"],
            "missing_directory_count": len(scan["missing_directories"]),
            "missing_directories": scan["missing_directories"],
        },
    }


def create_chart(class_counts):
    """Create and overwrite a grouped bar chart for the three splits."""
    positions = list(range(len(CLASSES)))
    bar_width = 0.25
    colors = ("#2563eb", "#16a34a", "#ea580c")

    figure, axis = plt.subplots(figsize=(10, 6))
    for index, split in enumerate(SPLITS):
        offsets = [position + (index - 1) * bar_width for position in positions]
        values = [class_counts[split][class_name] for class_name in CLASSES]
        bars = axis.bar(offsets, values, bar_width, label=split.capitalize(), color=colors[index])
        axis.bar_label(bars, padding=3)

    axis.set_title("SmartDocs AI Class Distribution")
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of Images")
    axis.set_xticks(positions)
    axis.set_xticklabels(CLASSES)
    axis.legend(title="Dataset split")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(CHART_FILE, dpi=150)
    plt.close(figure)


def print_report(report):
    """Print counts, percentages, balance, and output paths clearly."""
    print("\n" + "=" * 64)
    print("SMARTDOCS AI - CLASS DISTRIBUTION ANALYSIS")
    print("=" * 64)
    print(f"Dataset root             : {report['dataset_root']}")
    print(f"Total images             : {report['total_images']}")

    print("\nSplit counts:")
    for split, count in report["split_counts"].items():
        print(f"  {split:<22}: {count}")

    print("\nClass counts and percentages:")
    for split, counts in report["class_counts"].items():
        print(f"  {split}:")
        for class_name in CLASSES:
            percentage = report["class_percentages"][split][class_name]
            print(f"    {class_name:<18}: {counts[class_name]:>4} ({percentage:>6.2f}%)")

    print(f"\nBalance status           : {report['balance_status']['status']}")
    print(f"Unreadable files        : {report['file_quality']['unreadable_file_count']}")
    print(f"Missing directories      : {report['file_quality']['missing_directory_count']}")
    print(f"\nChart written to         : {CHART_FILE}")
    print(f"Report written to       : {REPORT_FILE}")


def main():
    """Scan, report, and chart the current dataset without modifying it."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_ROOT}")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    scan = scan_dataset()
    report = build_report(scan)
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    create_chart(scan["class_counts"])
    print_report(report)


if __name__ == "__main__":
    main()