"""Phase 6.3 - Image dimension and format analysis.

This script reads image metadata and file sizes from the final dataset. It is
strictly read-only and never resizes, converts, moves, or rewrites images.
"""

import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
REPORT_FILE = PROJECT_ROOT / "data" / "metadata" / "eda_image_properties.json"

SPLITS = ("train", "validation", "test")
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
    """Return the accumulator used for overall and split-level statistics."""
    return {
        "total_images": 0,
        "readable_images": 0,
        "widths": [],
        "heights": [],
        "dimensions": Counter(),
        "color_modes": Counter(),
        "file_extensions": Counter(),
        "file_sizes": [],
    }


def record_image(statistics, path):
    """Read one image's properties and add them to a statistics accumulator."""
    statistics["total_images"] += 1
    try:
        file_size = path.stat().st_size
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            color_mode = image.mode
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "filename": path.name,
            "error": str(exc),
        }

    statistics["readable_images"] += 1
    statistics["widths"].append(width)
    statistics["heights"].append(height)
    statistics["dimensions"][f"{width}x{height}"] += 1
    statistics["color_modes"][color_mode] += 1
    statistics["file_extensions"][path.suffix.lower()] += 1
    statistics["file_sizes"].append(file_size)
    return None


def scan_dataset():
    """Scan each split and collect readable properties and file issues."""
    split_statistics = {split: empty_statistics() for split in SPLITS}
    unreadable_images = []
    unsupported_files = []
    missing_splits = []

    for split in SPLITS:
        split_path = DATASET_ROOT / split
        if not split_path.is_dir():
            missing_splits.append(split_path.relative_to(PROJECT_ROOT).as_posix())
            continue

        for path in sorted(split_path.rglob("*")):
            if not path.is_file():
                continue
            extension = path.suffix.lower() or "[no extension]"
            if extension not in SUPPORTED_IMAGE_EXTENSIONS:
                unsupported_files.append(
                    {
                        "split": split,
                        "path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "extension": extension,
                    }
                )
                continue

            issue = record_image(split_statistics[split], path)
            if issue is not None:
                unreadable_images.append({"split": split, **issue})

    overall = empty_statistics()
    for split in SPLITS:
        split_data = split_statistics[split]
        overall["total_images"] += split_data["total_images"]
        overall["readable_images"] += split_data["readable_images"]
        overall["widths"].extend(split_data["widths"])
        overall["heights"].extend(split_data["heights"])
        overall["dimensions"].update(split_data["dimensions"])
        overall["color_modes"].update(split_data["color_modes"])
        overall["file_extensions"].update(split_data["file_extensions"])
        overall["file_sizes"].extend(split_data["file_sizes"])

    return {
        "overall": overall,
        "splits": split_statistics,
        "unreadable_images": unreadable_images,
        "unsupported_files": unsupported_files,
        "missing_splits": missing_splits,
    }


def serialize_statistics(statistics):
    """Convert an accumulator into the JSON report's public statistics shape."""
    widths = statistics["widths"]
    heights = statistics["heights"]
    file_sizes = statistics["file_sizes"]
    extensions = dict(sorted(statistics["file_extensions"].items()))
    dimensions = dict(sorted(statistics["dimensions"].items()))
    color_modes = dict(sorted(statistics["color_modes"].items()))

    return {
        "total_images": statistics["total_images"],
        "readable_images": statistics["readable_images"],
        "minimum_width": min(widths) if widths else None,
        "maximum_width": max(widths) if widths else None,
        "minimum_height": min(heights) if heights else None,
        "maximum_height": max(heights) if heights else None,
        "unique_image_dimensions": sorted(dimensions),
        "dimension_counts": dimensions,
        "unique_color_modes": sorted(color_modes),
        "color_mode_counts": color_modes,
        "file_extension_counts": extensions,
        "minimum_file_size_bytes": min(file_sizes) if file_sizes else None,
        "maximum_file_size_bytes": max(file_sizes) if file_sizes else None,
        "average_file_size_bytes": (
            round(sum(file_sizes) / len(file_sizes), 2) if file_sizes else None
        ),
    }


def build_checks(scan, serialized_statistics):
    """Create consistency and file-quality checks from observed data."""
    overall = serialized_statistics["overall"]
    extension_counts = overall["file_extension_counts"]
    dominant_extension = None
    if extension_counts:
        dominant_extension = sorted(
            extension_counts,
            key=lambda extension: (-extension_counts[extension], extension),
        )[0]

    unexpected_extensions = [
        extension
        for extension in extension_counts
        if extension != dominant_extension
    ]
    inconsistent_dimensions = len(overall["unique_image_dimensions"]) > 1
    inconsistent_color_modes = len(overall["unique_color_modes"]) > 1

    return {
        "unreadable_image_count": len(scan["unreadable_images"]),
        "unreadable_images": scan["unreadable_images"],
        "unsupported_file_count": len(scan["unsupported_files"]),
        "unsupported_files": scan["unsupported_files"],
        "missing_split_count": len(scan["missing_splits"]),
        "missing_splits": scan["missing_splits"],
        "inconsistent_dimensions": inconsistent_dimensions,
        "inconsistent_color_modes": inconsistent_color_modes,
        "dominant_file_extension": dominant_extension,
        "unexpected_file_extensions": unexpected_extensions,
        "has_unexpected_file_extensions": bool(unexpected_extensions),
    }


def build_report(scan):
    """Build the complete JSON report from the read-only scan."""
    serialized = {
        "overall": serialize_statistics(scan["overall"]),
        **{
            split: serialize_statistics(scan["splits"][split])
            for split in SPLITS
        },
    }
    return {
        "dataset_root": DATASET_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "overall_statistics": serialized["overall"],
        "train_statistics": serialized["train"],
        "validation_statistics": serialized["validation"],
        "test_statistics": serialized["test"],
        "validation_checks": build_checks(scan, serialized),
    }


def print_statistics(label, statistics):
    """Print one overall or split-level statistics section."""
    print(f"\n{label}:")
    print(f"  Total images           : {statistics['total_images']}")
    print(f"  Readable images        : {statistics['readable_images']}")
    print(
        f"  Width range            : {statistics['minimum_width']} - "
        f"{statistics['maximum_width']}"
    )
    print(
        f"  Height range           : {statistics['minimum_height']} - "
        f"{statistics['maximum_height']}"
    )
    print(f"  Unique dimensions      : {statistics['unique_image_dimensions']}")
    print(f"  Color modes            : {statistics['unique_color_modes']}")
    print(f"  Extensions             : {statistics['file_extension_counts']}")
    print(f"  Minimum file size      : {statistics['minimum_file_size_bytes']} bytes")
    print(f"  Maximum file size      : {statistics['maximum_file_size_bytes']} bytes")
    print(f"  Average file size      : {statistics['average_file_size_bytes']} bytes")


def print_report(report):
    """Print the complete analysis in a clear terminal format."""
    print("\n" + "=" * 68)
    print("SMARTDOCS AI - IMAGE DIMENSION & FORMAT ANALYSIS")
    print("=" * 68)
    print(f"Dataset root             : {report['dataset_root']}")
    print_statistics("Overall statistics", report["overall_statistics"])
    print_statistics("Train statistics", report["train_statistics"])
    print_statistics("Validation statistics", report["validation_statistics"])
    print_statistics("Test statistics", report["test_statistics"])

    checks = report["validation_checks"]
    print("\nValidation checks:")
    print(f"  Unreadable images      : {checks['unreadable_image_count']}")
    print(f"  Unsupported files      : {checks['unsupported_file_count']}")
    print(f"  Missing splits         : {checks['missing_split_count']}")
    print(f"  Inconsistent dimensions: {checks['inconsistent_dimensions']}")
    print(f"  Inconsistent color modes: {checks['inconsistent_color_modes']}")
    print(f"  Dominant extension     : {checks['dominant_file_extension']}")
    print(f"  Unexpected extensions  : {checks['unexpected_file_extensions']}")
    print(f"\nReport written to       : {REPORT_FILE}")


def main():
    """Scan the dataset and overwrite the repeatable properties report."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_ROOT}")

    scan = scan_dataset()
    report = build_report(scan)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print_report(report)


if __name__ == "__main__":
    main()