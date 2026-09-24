"""Phase 6.5 - Pixel-level statistics for the final dataset.

Images are read only. No dataset image is resized, converted, overwritten, or
otherwise modified by this analysis.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
REPORT_FILE = PROJECT_ROOT / "data" / "metadata" / "eda_pixel_statistics.json"

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

# Thresholds are expressed on the 8-bit RGB intensity scale [0, 255].
BLANK_INTENSITY = 0
DARK_MEAN_INTENSITY = 32
BRIGHT_MEAN_INTENSITY = 224


def empty_statistics():
    """Create accumulators for image and channel-level pixel statistics."""
    return {
        "total_images": 0,
        "readable_images": 0,
        "pixel_count": 0,
        "intensity_minimum": None,
        "intensity_maximum": None,
        "intensity_sum": 0.0,
        "intensity_squared_sum": 0.0,
        "channel_sum": np.zeros(3, dtype=np.float64),
        "channel_squared_sum": np.zeros(3, dtype=np.float64),
        "blank_image_count": 0,
        "dark_image_count": 0,
        "bright_image_count": 0,
    }


def add_image(statistics, path):
    """Read one image and add its RGB pixel values to the accumulators."""
    statistics["total_images"] += 1
    try:
        with Image.open(path) as image:
            # RGB conversion only creates an in-memory analysis array; the
            # source image on disk is never written or changed.
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "filename": path.name,
            "error": str(exc),
        }

    pixel_values = pixels.reshape(-1, 3)
    intensities = pixel_values.reshape(-1)
    image_mean = float(intensities.mean())
    statistics["readable_images"] += 1
    statistics["pixel_count"] += intensities.size
    statistics["intensity_minimum"] = (
        int(intensities.min())
        if statistics["intensity_minimum"] is None
        else min(statistics["intensity_minimum"], int(intensities.min()))
    )
    statistics["intensity_maximum"] = (
        int(intensities.max())
        if statistics["intensity_maximum"] is None
        else max(statistics["intensity_maximum"], int(intensities.max()))
    )
    statistics["intensity_sum"] += float(intensities.sum())
    statistics["intensity_squared_sum"] += float(np.square(intensities).sum())
    statistics["channel_sum"] += pixel_values.sum(axis=0)
    statistics["channel_squared_sum"] += np.square(pixel_values).sum(axis=0)

    statistics["blank_image_count"] += int(
        np.all(pixel_values == BLANK_INTENSITY)
    )
    statistics["dark_image_count"] += int(image_mean < DARK_MEAN_INTENSITY)
    statistics["bright_image_count"] += int(image_mean > BRIGHT_MEAN_INTENSITY)
    return None


def combine_statistics(target, source):
    """Merge one split accumulator into the overall accumulator."""
    target["total_images"] += source["total_images"]
    target["readable_images"] += source["readable_images"]
    target["pixel_count"] += source["pixel_count"]
    if source["intensity_minimum"] is not None:
        target["intensity_minimum"] = (
            source["intensity_minimum"]
            if target["intensity_minimum"] is None
            else min(target["intensity_minimum"], source["intensity_minimum"])
        )
        target["intensity_maximum"] = (
            source["intensity_maximum"]
            if target["intensity_maximum"] is None
            else max(target["intensity_maximum"], source["intensity_maximum"])
        )
    target["intensity_sum"] += source["intensity_sum"]
    target["intensity_squared_sum"] += source["intensity_squared_sum"]
    target["channel_sum"] += source["channel_sum"]
    target["channel_squared_sum"] += source["channel_squared_sum"]
    for key in ("blank_image_count", "dark_image_count", "bright_image_count"):
        target[key] += source[key]


def scan_dataset():
    """Scan each split and collect pixel statistics and readability issues."""
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
            issue = add_image(split_statistics[split], path)
            if issue is not None:
                unreadable_images.append({"split": split, **issue})

    overall = empty_statistics()
    for split in SPLITS:
        combine_statistics(overall, split_statistics[split])

    return {
        "overall": overall,
        "splits": split_statistics,
        "unreadable_images": unreadable_images,
        "unsupported_files": unsupported_files,
        "missing_splits": missing_splits,
    }


def population_standard_deviation(total, squared_total, count):
    """Calculate population standard deviation from aggregate moments."""
    if not count:
        return None
    variance = max((squared_total / count) - (total / count) ** 2, 0.0)
    return round(float(np.sqrt(variance)), 6)


def serialize_statistics(statistics):
    """Convert numeric accumulators into JSON-safe statistics."""
    pixel_count = statistics["pixel_count"]
    channel_count = pixel_count
    channel_mean = [
        round(float(value / channel_count), 6) if channel_count else None
        for value in statistics["channel_sum"]
    ]
    channel_std = [
        population_standard_deviation(
            statistics["channel_sum"][index],
            statistics["channel_squared_sum"][index],
            channel_count,
        )
        for index in range(3)
    ]
    return {
        "total_images": statistics["total_images"],
        "readable_images": statistics["readable_images"],
        "pixel_count": pixel_count,
        "pixel_intensity_minimum": statistics["intensity_minimum"],
        "pixel_intensity_maximum": statistics["intensity_maximum"],
        "mean_pixel_intensity": (
            round(statistics["intensity_sum"] / (pixel_count * 3), 6)
            if pixel_count
            else None
        ),
        "standard_deviation_pixel_intensity": population_standard_deviation(
            statistics["intensity_sum"],
            statistics["intensity_squared_sum"],
            pixel_count * 3,
        ),
        "rgb_channel_statistics": {
            "mean_R": channel_mean[0],
            "mean_G": channel_mean[1],
            "mean_B": channel_mean[2],
            "standard_deviation_R": channel_std[0],
            "standard_deviation_G": channel_std[1],
            "standard_deviation_B": channel_std[2],
        },
        "blank_image_count": statistics["blank_image_count"],
        "dark_image_count": statistics["dark_image_count"],
        "bright_image_count": statistics["bright_image_count"],
    }


def build_report(scan):
    """Build the complete JSON report from observed pixel values."""
    return {
        "dataset_root": DATASET_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "overall_statistics": serialize_statistics(scan["overall"]),
        "train_statistics": serialize_statistics(scan["splits"]["train"]),
        "validation_statistics": serialize_statistics(
            scan["splits"]["validation"]
        ),
        "test_statistics": serialize_statistics(scan["splits"]["test"]),
        "thresholds_used": {
            "pixel_scale": "8-bit RGB intensity, 0-255",
            "blank_image": "all RGB pixel values equal 0",
            "dark_image_mean_intensity_less_than": DARK_MEAN_INTENSITY,
            "bright_image_mean_intensity_greater_than": BRIGHT_MEAN_INTENSITY,
        },
        "processing_checks": {
            "unreadable_image_count": len(scan["unreadable_images"]),
            "unreadable_images": scan["unreadable_images"],
            "unsupported_file_count": len(scan["unsupported_files"]),
            "unsupported_files": scan["unsupported_files"],
            "missing_split_count": len(scan["missing_splits"]),
            "missing_splits": scan["missing_splits"],
            "images_converted_in_memory_to_rgb": True,
            "dataset_images_modified": False,
        },
    }


def print_statistics(label, statistics):
    """Print one overall or split-level pixel statistics section."""
    print(f"\n{label}:")
    print(f"  Total/readable images  : {statistics['total_images']} / {statistics['readable_images']}")
    print(f"  Pixel intensity min/max: {statistics['pixel_intensity_minimum']} / {statistics['pixel_intensity_maximum']}")
    print(f"  Mean pixel intensity   : {statistics['mean_pixel_intensity']}")
    print(f"  Pixel intensity stddev : {statistics['standard_deviation_pixel_intensity']}")
    print(f"  RGB means (R/G/B)      : {statistics['rgb_channel_statistics']['mean_R']} / {statistics['rgb_channel_statistics']['mean_G']} / {statistics['rgb_channel_statistics']['mean_B']}")
    print(f"  RGB stddevs (R/G/B)    : {statistics['rgb_channel_statistics']['standard_deviation_R']} / {statistics['rgb_channel_statistics']['standard_deviation_G']} / {statistics['rgb_channel_statistics']['standard_deviation_B']}")
    print(f"  Blank/dark/bright      : {statistics['blank_image_count']} / {statistics['dark_image_count']} / {statistics['bright_image_count']}")


def print_report(report):
    """Print the complete pixel analysis in a clear terminal format."""
    print("\n" + "=" * 72)
    print("SMARTDOCS AI - PIXEL / IMAGE STATISTICS")
    print("=" * 72)
    print(f"Dataset root             : {report['dataset_root']}")
    print_statistics("Overall statistics", report["overall_statistics"])
    print_statistics("Train statistics", report["train_statistics"])
    print_statistics("Validation statistics", report["validation_statistics"])
    print_statistics("Test statistics", report["test_statistics"])
    print("\nThresholds used:")
    for name, value in report["thresholds_used"].items():
        print(f"  {name:<34}: {value}")
    checks = report["processing_checks"]
    print("\nProcessing checks:")
    print(f"  Unreadable images      : {checks['unreadable_image_count']}")
    print(f"  Unsupported files      : {checks['unsupported_file_count']}")
    print(f"  Missing splits         : {checks['missing_split_count']}")
    print(f"\nReport written to       : {REPORT_FILE}")


def main():
    """Scan the dataset and overwrite the repeatable pixel report."""
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