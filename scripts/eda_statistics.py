"""Phase 6.1 - Dataset statistics and basic exploratory data analysis.

This script performs a read-only scan of the final ML-ready dataset. It does
not modify images, labels, or split assignments.
"""

import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
REPORT_FILE = PROJECT_ROOT / "data" / "metadata" / "eda_statistics.json"

SPLITS = ("train", "validation", "test")
CATEGORIES = ("invoices", "forms", "resumes", "other")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def scan_dataset():
    """Collect counts and image properties without changing dataset files."""
    split_counts = {split: 0 for split in SPLITS}
    category_counts = {
        split: {category: 0 for category in CATEGORIES} for split in SPLITS
    }
    dimensions = Counter()
    color_modes = Counter()
    extension_counts = Counter()
    unreadable_images = []
    unsupported_files = []

    for split in SPLITS:
        for category in CATEGORIES:
            directory = DATASET_ROOT / split / category
            if not directory.is_dir():
                continue

            for path in sorted(directory.iterdir()):
                if not path.is_file():
                    continue

                extension = path.suffix.lower() or "[no extension]"
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    unsupported_files.append(
                        {
                            "path": path.relative_to(PROJECT_ROOT).as_posix(),
                            "extension": extension,
                        }
                    )
                    continue

                split_counts[split] += 1
                category_counts[split][category] += 1
                extension_counts[extension] += 1
                location = {
                    "split": split,
                    "category": category,
                    "filename": path.name,
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                }

                try:
                    # Loading pixel data catches truncated or unreadable images.
                    with Image.open(path) as image:
                        image.load()
                        dimensions[tuple(image.size)] += 1
                        color_modes[image.mode] += 1
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    unreadable_images.append({**location, "error": str(exc)})

    return {
        "split_counts": split_counts,
        "category_counts": category_counts,
        "dimensions": dimensions,
        "color_modes": color_modes,
        "extension_counts": extension_counts,
        "unreadable_images": unreadable_images,
        "unsupported_files": unsupported_files,
    }


def build_report(scan):
    """Build a JSON-serializable report from the scan results."""
    dimensions = scan["dimensions"]
    all_sizes = sorted(dimensions)
    total_images = sum(scan["split_counts"].values())
    category_totals = {
        category: sum(
            scan["category_counts"][split][category] for split in SPLITS
        )
        for category in CATEGORIES
    }
    split_categories_balanced = all(
        len(set(scan["category_counts"][split].values())) == 1
        for split in SPLITS
    )
    category_totals_balanced = len(set(category_totals.values())) <= 1

    return {
        "dataset_root": DATASET_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "total_images": total_images,
        "split_counts": scan["split_counts"],
        "category_counts": scan["category_counts"],
        "category_totals": category_totals,
        "category_distribution_by_split": {
            split: {
                category: (
                    scan["category_counts"][split][category]
                    / scan["split_counts"][split]
                    if scan["split_counts"][split]
                    else 0
                )
                for category in CATEGORIES
            }
            for split in SPLITS
        },
        "image_properties": {
            "minimum_width": min((size[0] for size in all_sizes), default=None),
            "maximum_width": max((size[0] for size in all_sizes), default=None),
            "minimum_height": min((size[1] for size in all_sizes), default=None),
            "maximum_height": max((size[1] for size in all_sizes), default=None),
            "unique_image_sizes": [list(size) for size in all_sizes],
            "unique_color_modes": sorted(scan["color_modes"]),
            "file_extension_counts": dict(sorted(scan["extension_counts"].items())),
        },
        "balance": {
            "is_balanced": category_totals_balanced and split_categories_balanced,
            "category_totals_balanced": category_totals_balanced,
            "categories_balanced_within_each_split": split_categories_balanced,
        },
        "image_quality": {
            "unreadable_image_count": len(scan["unreadable_images"]),
            "unreadable_images": scan["unreadable_images"],
            "unsupported_file_count": len(scan["unsupported_files"]),
            "unsupported_files": scan["unsupported_files"],
        },
    }


def print_report(report):
    """Print the main findings in a readable terminal format."""
    print("\n" + "=" * 64)
    print("SMARTDOCS AI - DATASET EDA STATISTICS")
    print("=" * 64)
    print(f"Dataset root             : {report['dataset_root']}")
    print(f"Total images             : {report['total_images']}")
    print("\nImages by split:")
    for split, count in report["split_counts"].items():
        print(f"  {split:<22}: {count}")
    print("\nCategory counts by split:")
    for split, counts in report["category_counts"].items():
        print(f"  {split:<12}: {counts}")
    print(f"Category totals          : {report['category_totals']}")
    print("\nImage properties:")
    properties = report["image_properties"]
    print(
        f"  Width                  : {properties['minimum_width']} - "
        f"{properties['maximum_width']}"
    )
    print(
        f"  Height                 : {properties['minimum_height']} - "
        f"{properties['maximum_height']}"
    )
    print(f"  Unique image sizes     : {properties['unique_image_sizes']}")
    print(f"  Unique color modes     : {properties['unique_color_modes']}")
    print(f"  File extensions        : {properties['file_extension_counts']}")
    print(f"\nDataset balanced        : {report['balance']['is_balanced']}")
    print(
        "Unreadable images       : "
        f"{report['image_quality']['unreadable_image_count']}"
    )
    print(
        "Unsupported files       : "
        f"{report['image_quality']['unsupported_file_count']}"
    )
    print(f"\nReport written to       : {REPORT_FILE}")


def main():
    """Scan the dataset and overwrite the repeatable EDA report."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_ROOT}")

    report = build_report(scan_dataset())
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print_report(report)


if __name__ == "__main__":
    main()