"""Phase 6.6 - Descriptive comparison of document classes.

This analysis reads the final dataset only. It does not modify images, labels,
split assignments, or any other dataset content.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
REPORT_FILE = METADATA_DIR / "eda_classwise_comparison.json"
INTENSITY_CHART = METADATA_DIR / "classwise_mean_intensity.png"
FILE_SIZE_CHART = METADATA_DIR / "classwise_file_size.png"

SPLITS = ("train", "validation", "test")
CLASSES = ("invoices", "forms", "resumes", "other")
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def empty_statistics():
    """Create aggregate accumulators for one class."""
    return {
        "image_count": 0,
        "readable_image_count": 0,
        "pixel_count": 0,
        "intensity_sum": 0.0,
        "intensity_squared_sum": 0.0,
        "channel_sum": np.zeros(3, dtype=np.float64),
        "file_sizes": [],
    }


def add_image(statistics, path):
    """Read one image and add its pixel and file-size values."""
    statistics["image_count"] += 1
    try:
        file_size = path.stat().st_size
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "filename": path.name,
            "error": str(exc),
        }

    values = pixels.reshape(-1, 3)
    intensities = values.reshape(-1)
    statistics["readable_image_count"] += 1
    statistics["pixel_count"] += intensities.size
    statistics["intensity_sum"] += float(intensities.sum())
    statistics["intensity_squared_sum"] += float(np.square(intensities).sum())
    statistics["channel_sum"] += values.sum(axis=0)
    statistics["file_sizes"].append(file_size)
    return None


def scan_dataset():
    """Collect class-level values and file processing issues."""
    class_statistics = {class_name: empty_statistics() for class_name in CLASSES}
    unreadable_images = []
    unsupported_files = []
    missing_directories = []

    for split in SPLITS:
        for class_name in CLASSES:
            directory = DATASET_ROOT / split / class_name
            if not directory.is_dir():
                missing_directories.append(
                    directory.relative_to(PROJECT_ROOT).as_posix()
                )
                continue

            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                extension = path.suffix.lower() or "[no extension]"
                if extension not in SUPPORTED_IMAGE_EXTENSIONS:
                    unsupported_files.append(
                        {
                            "split": split,
                            "class": class_name,
                            "path": path.relative_to(PROJECT_ROOT).as_posix(),
                            "extension": extension,
                        }
                    )
                    continue
                issue = add_image(class_statistics[class_name], path)
                if issue is not None:
                    unreadable_images.append(
                        {"split": split, "class": class_name, **issue}
                    )

    return {
        "class_statistics": class_statistics,
        "unreadable_images": unreadable_images,
        "unsupported_files": unsupported_files,
        "missing_directories": missing_directories,
    }


def population_standard_deviation(total, squared_total, count):
    """Calculate population standard deviation from aggregate pixel moments."""
    if not count:
        return None
    variance = max((squared_total / count) - (total / count) ** 2, 0.0)
    return round(float(np.sqrt(variance)), 6)


def serialize_statistics(statistics):
    """Convert class accumulators to JSON-safe descriptive statistics."""
    pixel_count = statistics["pixel_count"]
    image_count = statistics["readable_image_count"]
    file_sizes = statistics["file_sizes"]
    return {
        "image_count": statistics["image_count"],
        "readable_image_count": image_count,
        "average_pixel_intensity": (
            round(statistics["intensity_sum"] / (pixel_count * 3), 6)
            if pixel_count
            else None
        ),
        "pixel_intensity_standard_deviation": population_standard_deviation(
            statistics["intensity_sum"],
            statistics["intensity_squared_sum"],
            pixel_count * 3,
        ),
        "average_red_channel_intensity": (
            round(float(statistics["channel_sum"][0] / pixel_count), 6)
            if pixel_count
            else None
        ),
        "average_green_channel_intensity": (
            round(float(statistics["channel_sum"][1] / pixel_count), 6)
            if pixel_count
            else None
        ),
        "average_blue_channel_intensity": (
            round(float(statistics["channel_sum"][2] / pixel_count), 6)
            if pixel_count
            else None
        ),
        "average_file_size_bytes": (
            round(sum(file_sizes) / len(file_sizes), 2) if file_sizes else None
        ),
        "minimum_file_size_bytes": min(file_sizes) if file_sizes else None,
        "maximum_file_size_bytes": max(file_sizes) if file_sizes else None,
    }


def extrema(statistics, key, mode):
    """Return all class names sharing the minimum or maximum value."""
    values = {
        class_name: statistics[class_name][key]
        for class_name in CLASSES
        if statistics[class_name][key] is not None
    }
    if not values:
        return []
    target = min(values.values()) if mode == "minimum" else max(values.values())
    return [class_name for class_name in CLASSES if values[class_name] == target]


def build_report(scan):
    """Build the class statistics, comparisons, and processing checks."""
    class_statistics = {
        class_name: serialize_statistics(scan["class_statistics"][class_name])
        for class_name in CLASSES
    }
    return {
        "dataset_root": DATASET_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "classes_analyzed": list(CLASSES),
        "class_statistics": class_statistics,
        "comparison_results": {
            "class_with_highest_average_brightness": extrema(
                class_statistics, "average_pixel_intensity", "maximum"
            ),
            "class_with_lowest_average_brightness": extrema(
                class_statistics, "average_pixel_intensity", "minimum"
            ),
            "class_with_largest_average_file_size": extrema(
                class_statistics, "average_file_size_bytes", "maximum"
            ),
            "class_with_smallest_average_file_size": extrema(
                class_statistics, "average_file_size_bytes", "minimum"
            ),
        },
        "processing_checks": {
            "unreadable_image_count": len(scan["unreadable_images"]),
            "unreadable_images": scan["unreadable_images"],
            "unsupported_file_count": len(scan["unsupported_files"]),
            "unsupported_files": scan["unsupported_files"],
            "missing_directory_count": len(scan["missing_directories"]),
            "missing_directories": scan["missing_directories"],
            "dataset_images_modified": False,
        },
    }


def save_bar_chart(path, title, ylabel, values):
    """Create and overwrite a simple class comparison bar chart."""
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(CLASSES, values, color=("#2563eb", "#16a34a", "#ea580c", "#7c3aed"))
    axis.bar_label(bars, padding=3, fmt="%.2f")
    axis.set_title(title)
    axis.set_xlabel("Class")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def create_visualizations(class_statistics):
    """Create the requested brightness and file-size visualizations."""
    intensities = [
        class_statistics[class_name]["average_pixel_intensity"] or 0
        for class_name in CLASSES
    ]
    file_sizes = [
        class_statistics[class_name]["average_file_size_bytes"] or 0
        for class_name in CLASSES
    ]
    save_bar_chart(
        INTENSITY_CHART,
        "Average Pixel Intensity by Class",
        "Average Pixel Intensity",
        intensities,
    )
    save_bar_chart(
        FILE_SIZE_CHART,
        "Average File Size by Class",
        "Average File Size (bytes)",
        file_sizes,
    )


def print_report(report):
    """Print class statistics and descriptive comparison results."""
    print("\n" + "=" * 72)
    print("SMARTDOCS AI - CLASS-WISE COMPARISON")
    print("=" * 72)
    print(f"Dataset root             : {report['dataset_root']}")
    print("\nClass statistics:")
    for class_name in report["classes_analyzed"]:
        statistics = report["class_statistics"][class_name]
        print(f"  {class_name}:")
        print(f"    Images/readable       : {statistics['image_count']} / {statistics['readable_image_count']}")
        print(f"    Mean intensity        : {statistics['average_pixel_intensity']}")
        print(f"    Intensity stddev      : {statistics['pixel_intensity_standard_deviation']}")
        print(f"    Mean RGB (R/G/B)      : {statistics['average_red_channel_intensity']} / {statistics['average_green_channel_intensity']} / {statistics['average_blue_channel_intensity']}")
        print(f"    File size avg/min/max : {statistics['average_file_size_bytes']} / {statistics['minimum_file_size_bytes']} / {statistics['maximum_file_size_bytes']} bytes")

    print("\nComparison results (descriptive only):")
    for name, classes in report["comparison_results"].items():
        print(f"  {name:<46}: {', '.join(classes) if classes else 'N/A'}")
    checks = report["processing_checks"]
    print("\nProcessing checks:")
    print(f"  Unreadable images      : {checks['unreadable_image_count']}")
    print(f"  Unsupported files      : {checks['unsupported_file_count']}")
    print(f"  Missing directories    : {checks['missing_directory_count']}")
    print(f"\nCharts written to       : {INTENSITY_CHART}")
    print(f"                         {FILE_SIZE_CHART}")
    print(f"Report written to       : {REPORT_FILE}")


def main():
    """Scan, chart, and report the current class-level dataset properties."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_ROOT}")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(scan_dataset())
    create_visualizations(report["class_statistics"])
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print_report(report)


if __name__ == "__main__":
    main()