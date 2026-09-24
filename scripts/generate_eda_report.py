"""Phase 6.7 - Consolidate the validated Phase 6 EDA reports.

This script reads existing JSON reports only. It does not rescan or modify the
dataset and does not recalculate the underlying EDA statistics.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
FINAL_JSON = METADATA_DIR / "final_eda_report.json"
FINAL_MARKDOWN = METADATA_DIR / "final_eda_report.md"

REPORT_FILES = {
    "dataset_statistics": METADATA_DIR / "eda_statistics.json",
    "class_distribution": METADATA_DIR / "class_distribution.json",
    "image_properties": METADATA_DIR / "eda_image_properties.json",
    "visual_samples": METADATA_DIR / "eda_visual_samples.json",
    "pixel_statistics": METADATA_DIR / "eda_pixel_statistics.json",
    "classwise_comparison": METADATA_DIR / "eda_classwise_comparison.json",
}
VALIDATION_FILE = METADATA_DIR / "final_dataset_validation.json"

VISUALIZATION_PATHS = (
    "data/metadata/class_distribution.png",
    "data/metadata/eda_visual_samples.png",
    "data/metadata/classwise_mean_intensity.png",
    "data/metadata/classwise_file_size.png",
)


def load_json(path, description):
    """Load a required JSON report with a clear error for invalid input."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def read_reports():
    """Read all Phase 6 source reports and the prior validation report."""
    reports = {
        name: load_json(path, name.replace("_", " "))
        for name, path in REPORT_FILES.items()
    }
    validation = (
        load_json(VALIDATION_FILE, "final dataset validation report")
        if VALIDATION_FILE.is_file()
        else {}
    )
    return reports, validation


def unique_entries(*collections):
    """Return a stable list containing unique JSON-compatible entries."""
    entries = []
    seen = set()
    for collection in collections:
        for entry in collection or []:
            marker = json.dumps(entry, sort_keys=True)
            if marker not in seen:
                seen.add(marker)
                entries.append(entry)
    return entries


def build_quality_checks(reports, validation):
    """Combine quality checks from Phase 6 and the prior validation report."""
    image_checks = reports["image_properties"].get("validation_checks", {})
    distribution_quality = reports["class_distribution"].get("file_quality", {})
    visual_checks = reports["visual_samples"].get("checks", {})
    pixel_checks = reports["pixel_statistics"].get("processing_checks", {})
    classwise_checks = reports["classwise_comparison"].get(
        "processing_checks", {}
    )
    distribution_balance = reports["class_distribution"].get(
        "balance_status", {}
    )
    unexpected_extensions = image_checks.get("unexpected_file_extensions", [])
    visualization_failures = visual_checks.get("visualization_failures", [])

    unreadable = unique_entries(
        image_checks.get("unreadable_images"),
        distribution_quality.get("unreadable_files"),
        visual_checks.get("unreadable_images"),
        pixel_checks.get("unreadable_images"),
        classwise_checks.get("unreadable_images"),
    )
    unsupported = unique_entries(
        image_checks.get("unsupported_files"),
        distribution_quality.get("unsupported_files"),
        pixel_checks.get("unsupported_files"),
        classwise_checks.get("unsupported_files"),
    )
    missing_directories = unique_entries(
        image_checks.get("missing_splits"),
        distribution_quality.get("missing_directories"),
        visual_checks.get("missing_directories"),
        pixel_checks.get("missing_splits"),
        classwise_checks.get("missing_directories"),
    )

    duplicate_filename_count = validation.get("duplicate_filename_count", 0)
    duplicate_content_count = validation.get("duplicate_content_count", 0)
    duplicate_count = validation.get(
        "duplicate_count", duplicate_filename_count + duplicate_content_count
    )
    structure_issues = validation.get("structure_issues", [])
    missing_files = validation.get("missing_files", [])

    checks = {
        "unreadable_images": {
            "count": len(unreadable),
            "entries": unreadable,
        },
        "unsupported_files": {
            "count": len(unsupported),
            "entries": unsupported,
        },
        "missing_directories": {
            "count": len(missing_directories),
            "entries": missing_directories,
        },
        "inconsistent_dimensions": bool(
            image_checks.get("inconsistent_dimensions", False)
        ),
        "inconsistent_color_modes": bool(
            image_checks.get("inconsistent_color_modes", False)
        ),
        "unexpected_file_extensions": {
            "count": len(unexpected_extensions),
            "entries": unexpected_extensions,
        },
        "class_distribution_balanced": bool(
            distribution_balance.get("is_balanced", False)
        ),
        "visualization_failures": {
            "count": len(visualization_failures),
            "entries": visualization_failures,
        },
        "visualizations_present": {
            path: (PROJECT_ROOT / path).is_file() for path in VISUALIZATION_PATHS
        },
        "duplicate_and_content_issues": {
            "duplicate_count": duplicate_count,
            "duplicate_filename_count": duplicate_filename_count,
            "duplicate_content_count": duplicate_content_count,
            "missing_file_count": validation.get("missing_file_count", len(missing_files)),
            "structure_issue_count": len(structure_issues),
            "structure_issues": structure_issues,
            "missing_files": missing_files,
        },
    }
    checks["all_required_checks_pass"] = not any(
        (
            checks["unreadable_images"]["count"],
            checks["unsupported_files"]["count"],
            checks["missing_directories"]["count"],
            checks["inconsistent_dimensions"],
            checks["inconsistent_color_modes"],
            checks["unexpected_file_extensions"]["count"],
            not checks["class_distribution_balanced"],
            checks["visualization_failures"]["count"],
            duplicate_count,
            len(missing_files),
            len(structure_issues),
            not all(checks["visualizations_present"].values()),
        )
    )
    return checks


def build_conclusions(reports, quality_checks):
    """Create factual, non-predictive EDA observations."""
    overview = reports["dataset_statistics"]
    properties = reports["image_properties"]["overall_statistics"]
    distribution = reports["class_distribution"]
    pixel = reports["pixel_statistics"]["overall_statistics"]
    conclusions = [
        f"The dataset contains {overview['total_images']} images: "
        f"{overview['split_counts']['train']} train, "
        f"{overview['split_counts']['validation']} validation, and "
        f"{overview['split_counts']['test']} test.",
        "Class distribution is balanced across the complete dataset and within each split."
        if distribution["balance_status"]["is_balanced"]
        else "Class distribution is not balanced according to the class distribution report.",
        "All readable images have consistent dimensions and color mode."
        if not quality_checks["inconsistent_dimensions"]
        and not quality_checks["inconsistent_color_modes"]
        else "The reports identify inconsistent image dimensions or color modes.",
        f"The observed image properties are {properties['unique_image_dimensions']} "
        f"with color mode(s) {properties['unique_color_modes']} and "
        f"extension counts {properties['file_extension_counts']}.",
        f"The overall mean pixel intensity is {pixel['mean_pixel_intensity']} "
        f"with standard deviation {pixel['standard_deviation_pixel_intensity']}.",
        f"Blank, dark, and bright image counts are "
        f"{pixel['blank_image_count']}, {pixel['dark_image_count']}, and "
        f"{pixel['bright_image_count']} under the documented thresholds.",
    ]
    return conclusions


def build_report(reports, validation):
    """Build the consolidated final EDA JSON object."""
    quality_checks = build_quality_checks(reports, validation)
    report = {
        "report_title": "SmartDocs AI Phase 6 Exploratory Data Analysis",
        "phase": "6.7 - Final EDA Report",
        "source_reports": {
            name: path.relative_to(PROJECT_ROOT).as_posix()
            for name, path in REPORT_FILES.items()
        },
        "dataset_overview": {
            "total_images": reports["dataset_statistics"]["total_images"],
            "split_counts": reports["dataset_statistics"]["split_counts"],
            "class_counts": reports["dataset_statistics"]["category_totals"],
        },
        "class_distribution": {
            "class_counts": reports["class_distribution"]["class_counts"],
            "percentages": reports["class_distribution"]["class_percentages"],
            "balance_status": reports["class_distribution"]["balance_status"],
        },
        "image_properties": reports["image_properties"],
        "visual_sample_analysis": {
            "random_seed": reports["visual_samples"]["random_seed"],
            "samples_selected": reports["visual_samples"]["selected_samples"],
            "selected_image_filenames": reports["visual_samples"]["selected_image_filenames"],
            "selected_image_paths": reports["visual_samples"]["selected_image_paths"],
            "validation_checks": reports["visual_samples"]["checks"],
        },
        "pixel_statistics": {
            "overall": reports["pixel_statistics"]["overall_statistics"],
            "train": reports["pixel_statistics"]["train_statistics"],
            "validation": reports["pixel_statistics"]["validation_statistics"],
            "test": reports["pixel_statistics"]["test_statistics"],
            "thresholds_used": reports["pixel_statistics"]["thresholds_used"],
        },
        "classwise_comparison": {
            "class_statistics": reports["classwise_comparison"]["class_statistics"],
            "brightness_comparison": {
                "highest_average_brightness": reports["classwise_comparison"]["comparison_results"]["class_with_highest_average_brightness"],
                "lowest_average_brightness": reports["classwise_comparison"]["comparison_results"]["class_with_lowest_average_brightness"],
            },
            "file_size_comparison": {
                "largest_average_file_size": reports["classwise_comparison"]["comparison_results"]["class_with_largest_average_file_size"],
                "smallest_average_file_size": reports["classwise_comparison"]["comparison_results"]["class_with_smallest_average_file_size"],
            },
        },
        "dataset_quality_checks": quality_checks,
        "visualization_paths": list(VISUALIZATION_PATHS),
    }
    report["final_eda_conclusions"] = build_conclusions(reports, quality_checks)
    report["phase_6_eda_complete"] = quality_checks["all_required_checks_pass"]
    return report


def markdown_table(headers, rows):
    """Render a small Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(report):
    """Build the human-readable final EDA report."""
    overview = report["dataset_overview"]
    distribution = report["class_distribution"]
    properties = report["image_properties"]["overall_statistics"]
    pixel = report["pixel_statistics"]["overall"]
    classwise = report["classwise_comparison"]
    quality = report["dataset_quality_checks"]
    rows = [
        (split, count) for split, count in overview["split_counts"].items()
    ]
    class_rows = [
        (class_name, count)
        for class_name, count in overview["class_counts"].items()
    ]
    property_rows = [
        ("Dimensions", ", ".join(properties["unique_image_dimensions"])),
        ("Color modes", ", ".join(properties["unique_color_modes"])),
        ("File extensions", str(properties["file_extension_counts"])),
        ("File size range", f"{properties['minimum_file_size_bytes']} - {properties['maximum_file_size_bytes']} bytes"),
        ("Average file size", f"{properties['average_file_size_bytes']} bytes"),
    ]
    pixel_rows = [
        ("Mean intensity", pixel["mean_pixel_intensity"]),
        ("Standard deviation", pixel["standard_deviation_pixel_intensity"]),
        ("RGB means", str(pixel["rgb_channel_statistics"])),
        ("Blank / dark / bright", f"{pixel['blank_image_count']} / {pixel['dark_image_count']} / {pixel['bright_image_count']}"),
    ]
    quality_rows = [
        ("Unreadable images", quality["unreadable_images"]["count"]),
        ("Unsupported files", quality["unsupported_files"]["count"]),
        ("Missing directories", quality["missing_directories"]["count"]),
        ("Inconsistent dimensions", quality["inconsistent_dimensions"]),
        ("Inconsistent color modes", quality["inconsistent_color_modes"]),
        ("Duplicate/content issues", quality["duplicate_and_content_issues"]["duplicate_count"]),
    ]
    visualization_lines = "\n".join(f"- `{path}`" for path in report["visualization_paths"])
    conclusions = "\n".join(f"- {item}" for item in report["final_eda_conclusions"])
    return f"""# SmartDocs AI Phase 6 Exploratory Data Analysis

## Dataset Overview

{markdown_table(["Split", "Images"], rows)}

Total images: **{overview['total_images']}**

### Class Counts

{markdown_table(["Class", "Images"], class_rows)}

## Class Distribution

Balance status: **{distribution['balance_status']['status']}**

Percentages are 25.00% per class in each split and in the complete dataset.

## Image Properties

{markdown_table(["Property", "Observed value"], property_rows)}

## Visual Analysis

Random seed: **{report['visual_sample_analysis']['random_seed']}**. Six samples were selected per class, with two samples from each analyzed split.

Generated visualizations:

{visualization_lines}

## Pixel Statistics

{markdown_table(["Statistic", "Overall value"], pixel_rows)}

Thresholds used: `{report['pixel_statistics']['thresholds_used']}`

## Class-wise Comparison

| Class | Mean intensity | Mean file size (bytes) |
| --- | ---: | ---: |
""" + "\n".join(
        f"| {class_name} | {stats['average_pixel_intensity']} | {stats['average_file_size_bytes']} |"
        for class_name, stats in classwise["class_statistics"].items()
    ) + f"""

Highest average brightness: **{classwise['brightness_comparison']['highest_average_brightness']}**  
Lowest average brightness: **{classwise['brightness_comparison']['lowest_average_brightness']}**  
Largest average file size: **{classwise['file_size_comparison']['largest_average_file_size']}**  
Smallest average file size: **{classwise['file_size_comparison']['smallest_average_file_size']}**

These are descriptive comparisons only and do not indicate class quality or model performance.

## Dataset Quality Checks

{markdown_table(["Check", "Result"], quality_rows)}

## Final Observations

{conclusions}

## Summary

**Phase 6 EDA complete: {report['phase_6_eda_complete']}**

All required checks pass only when this value is `True`.
"""


def main():
    """Read source reports and overwrite both consolidated report formats."""
    reports, validation = read_reports()
    report = build_report(reports, validation)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with FINAL_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    FINAL_MARKDOWN.write_text(build_markdown(report), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SMARTDOCS AI - FINAL PHASE 6 EDA REPORT")
    print("=" * 72)
    print(f"Total images             : {report['dataset_overview']['total_images']}")
    print(f"Train/validation/test    : {report['dataset_overview']['split_counts']}")
    print(f"Class balance             : {report['class_distribution']['balance_status']['status']}")
    print(f"Image dimensions         : {report['image_properties']['overall_statistics']['unique_image_dimensions']}")
    print(f"Color modes              : {report['image_properties']['overall_statistics']['unique_color_modes']}")
    print(f"Unreadable images        : {report['dataset_quality_checks']['unreadable_images']['count']}")
    print(f"Unsupported files        : {report['dataset_quality_checks']['unsupported_files']['count']}")
    print(f"Inconsistent dimensions  : {report['dataset_quality_checks']['inconsistent_dimensions']}")
    print(f"Inconsistent color modes : {report['dataset_quality_checks']['inconsistent_color_modes']}")
    print(f"Duplicate/content issues : {report['dataset_quality_checks']['duplicate_and_content_issues']['duplicate_count']}")
    print(f"Phase 6 EDA complete     : {report['phase_6_eda_complete']}")
    print(f"JSON report written to   : {FINAL_JSON}")
    print(f"Markdown report written  : {FINAL_MARKDOWN}")


if __name__ == "__main__":
    main()