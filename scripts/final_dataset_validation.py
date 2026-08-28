"""Phase 5.7 - Final ML-ready dataset validation."""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
MANIFEST_FILE = METADATA_DIR / "processed_dataset_manifest.json"
REPORT_FILE = METADATA_DIR / "final_dataset_validation.json"

SPLITS = ("train", "validation", "test")
CATEGORIES = ("invoices", "forms", "resumes", "other")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_SPLIT_COUNTS = {"train": 1400, "validation": 300, "test": 300}
EXPECTED_CATEGORY_COUNTS = {
    "train": {category: 350 for category in CATEGORIES},
    "validation": {category: 75 for category in CATEGORIES},
    "test": {category: 75 for category in CATEGORIES},
}
HASH_CHUNK_SIZE = 65536


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path, description):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {description}: {exc}") from exc


def validate_structure():
    issues = []
    for split in SPLITS:
        split_path = DATASET_ROOT / split
        if not split_path.is_dir():
            issues.append(f"Missing split directory: {split_path}")
        for category in CATEGORIES:
            category_path = split_path / category
            if not category_path.is_dir():
                issues.append(f"Missing category directory: {category_path}")
    return issues


def scan_dataset():
    split_counts = {split: 0 for split in SPLITS}
    category_counts = {
        split: {category: 0 for category in CATEGORIES} for split in SPLITS
    }
    invalid_images = []
    wrong_dimensions = []
    wrong_color_mode = []
    filename_locations = defaultdict(list)
    content_locations = defaultdict(list)
    all_images = []

    for split in SPLITS:
        for category in CATEGORIES:
            directory = DATASET_ROOT / split / category
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if not is_image_file(path):
                    continue
                split_counts[split] += 1
                category_counts[split][category] += 1
                location = {
                    "split": split,
                    "category": category,
                    "filename": path.name,
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                }
                all_images.append(path)
                filename_locations[path.name].append(location)
                try:
                    with Image.open(path) as image:
                        image.load()
                        if image.size != (224, 224):
                            wrong_dimensions.append({**location, "dimensions": list(image.size)})
                        if image.mode != "RGB":
                            wrong_color_mode.append({**location, "mode": image.mode})
                        content_locations[sha256_file(path)].append(location)
                except (OSError, ValueError) as exc:
                    invalid_images.append({**location, "error": str(exc)})

    duplicate_filenames = [
        {"filename": filename, "locations": locations}
        for filename, locations in filename_locations.items()
        if len({entry["split"] for entry in locations}) > 1
    ]
    duplicate_contents = [
        {"sha256": digest, "locations": locations}
        for digest, locations in content_locations.items()
        if len({entry["split"] for entry in locations}) > 1
    ]

    return {
        "all_images": all_images,
        "split_counts": split_counts,
        "category_counts": category_counts,
        "invalid_images": invalid_images,
        "wrong_dimensions": wrong_dimensions,
        "wrong_color_mode": wrong_color_mode,
        "duplicate_filenames": duplicate_filenames,
        "duplicate_contents": duplicate_contents,
    }


def check_manifest(dataset_paths):
    """Return manifest paths that do not exist in the final dataset."""
    try:
        manifest = load_json(MANIFEST_FILE, "processed dataset manifest")
    except (FileNotFoundError, RuntimeError) as exc:
        return [str(exc)]

    records = manifest.get("images", []) if isinstance(manifest, dict) else []
    missing = []
    for record in records:
        final_path = record.get("final_path") if isinstance(record, dict) else None
        if not final_path:
            missing.append("Manifest record has no final_path")
            continue
        path = PROJECT_ROOT / Path(final_path)
        if not path.is_file():
            missing.append(final_path)

    expected_paths = {path.resolve() for path in dataset_paths}
    manifest_paths = {
        (PROJECT_ROOT / Path(record["final_path"])).resolve()
        for record in records
        if isinstance(record, dict) and record.get("final_path")
    }
    missing.extend(
        f"Dataset image absent from manifest: {path.relative_to(PROJECT_ROOT).as_posix()}"
        for path in sorted(expected_paths - manifest_paths)
    )
    return sorted(set(missing))


def raw_inventory():
    """Capture a read-only fingerprint for future rerun comparisons."""
    if not RAW_ROOT.is_dir():
        return None, [f"Missing raw dataset directory: {RAW_ROOT}"]

    inventory = []
    issues = []
    for path in sorted(RAW_ROOT.rglob("*")):
        if not path.is_file():
            continue
        try:
            inventory.append({
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        except OSError as exc:
            issues.append(f"Could not read raw file {path}: {exc}")
    return inventory, issues


def compare_raw_inventory(current):
    """Compare against the prior report when one exists; never changes raw data."""
    if not REPORT_FILE.exists():
        return {"status": "BASELINE_CAPTURED", "changed_files": []}
    try:
        previous = load_json(REPORT_FILE, "previous validation report")
    except (FileNotFoundError, RuntimeError):
        return {"status": "BASELINE_UNAVAILABLE", "changed_files": []}

    previous_inventory = previous.get("raw_dataset", {}).get("inventory", [])
    if not previous_inventory:
        return {"status": "BASELINE_UNAVAILABLE", "changed_files": []}
    if previous_inventory == current:
        return {"status": "PASS", "changed_files": []}

    old = {entry["path"]: entry for entry in previous_inventory}
    new = {entry["path"]: entry for entry in current}
    changed = sorted(set(old) | set(new))
    changed = [path for path in changed if old.get(path) != new.get(path)]
    return {"status": "FAIL", "changed_files": changed}


def build_report(scan, missing_files, raw_data, raw_issues, raw_comparison):
    split_counts = scan["split_counts"]
    total_images = sum(split_counts.values())
    count_checks = {
        split: split_counts[split] == EXPECTED_SPLIT_COUNTS[split] for split in SPLITS
    }
    category_check = scan["category_counts"] == EXPECTED_CATEGORY_COUNTS
    structure_issues = validate_structure()
    status = (
        not structure_issues
        and all(count_checks.values())
        and total_images == 2000
        and category_check
        and not scan["invalid_images"]
        and not scan["wrong_dimensions"]
        and not scan["wrong_color_mode"]
        and not scan["duplicate_filenames"]
        and not scan["duplicate_contents"]
        and not missing_files
        and not raw_issues
        and raw_comparison["status"] != "FAIL"
    )

    return {
        "total_images": total_images,
        "split_counts": split_counts,
        "category_counts": scan["category_counts"],
        "invalid_corrupted_image_count": len(scan["invalid_images"]),
        "wrong_dimension_count": len(scan["wrong_dimensions"]),
        "wrong_color_mode_count": len(scan["wrong_color_mode"]),
        "duplicate_count": len(scan["duplicate_filenames"]) + len(scan["duplicate_contents"]),
        "duplicate_filename_count": len(scan["duplicate_filenames"]),
        "duplicate_content_count": len(scan["duplicate_contents"]),
        "missing_file_count": len(missing_files),
        "structure_issues": structure_issues,
        "invalid_corrupted_images": scan["invalid_images"],
        "wrong_dimensions": scan["wrong_dimensions"],
        "wrong_color_modes": scan["wrong_color_mode"],
        "duplicate_filenames": scan["duplicate_filenames"],
        "duplicate_contents": scan["duplicate_contents"],
        "missing_files": missing_files,
        "raw_dataset": {
            "status": raw_comparison["status"],
            "inventory": raw_data,
            "issues": raw_issues,
            "changed_files": raw_comparison["changed_files"],
        },
        "final_validation_status": "PASS" if status else "FAIL",
    }


def print_summary(report):
    print("\n" + "=" * 60)
    print("FINAL DATASET VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total images             : {report['total_images']}")
    for split in SPLITS:
        print(f"  {split.capitalize():<24}: {report['split_counts'][split]}")
    print(f"  Invalid/corrupted images : {report['invalid_corrupted_image_count']}")
    print(f"  Wrong dimensions         : {report['wrong_dimension_count']}")
    print(f"  Wrong color mode         : {report['wrong_color_mode_count']}")
    print(f"  Duplicate filenames      : {report['duplicate_filename_count']}")
    print(f"  Duplicate contents       : {report['duplicate_content_count']}")
    print(f"  Missing manifest files   : {report['missing_file_count']}")
    print(f"  Raw dataset check        : {report['raw_dataset']['status']}")
    print(f"\n  FINAL RESULT: {report['final_validation_status']}")
    print("=" * 60)


def main():
    print("SmartDocs AI - Phase 5.7: Final Dataset Validation")
    print("Reading dataset files only; no dataset files will be modified.")
    try:
        scan = scan_dataset()
        raw_data, raw_issues = raw_inventory()
        raw_comparison = compare_raw_inventory(raw_data or [])
        missing_files = check_manifest(scan["all_images"])
        report = build_report(
            scan, missing_files, raw_data or [], raw_issues, raw_comparison
        )
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        with REPORT_FILE.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
    except (OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_summary(report)
    return 0 if report["final_validation_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
