"""
Phase 5.6 – Final Processed Dataset Organization
==================================================
SmartDocs AI Project

Purpose
-------
Read the split assignments from data/metadata/dataset_split.json and copy
each processed image into the corresponding split + category folder under
data/processed/dataset/.

Final layout
------------
    data/processed/dataset/
        train/
            invoices/    (350 images)
            forms/       (350 images)
            resumes/     (350 images)
            other/       (350 images)
        validation/
            invoices/    ( 75 images)
            forms/       ( 75 images)
            resumes/     ( 75 images)
            other/       ( 75 images)
        test/
            invoices/    ( 75 images)
            forms/       ( 75 images)
            resumes/     ( 75 images)
            other/       ( 75 images)

Design decisions
----------------
* Copies are used instead of symbolic links because Windows symlinks require
  elevated privileges and are unreliable across different NTFS configurations.
  Additional disk usage is reported clearly in the summary.
* Source images in data/processed/images/ are NEVER modified or deleted.
* data/raw/ is NEVER touched.
* Idempotent: if a destination file already exists and its size matches the
  source, the copy is skipped (no duplicate work on re-runs).
* A manifest is written to data/metadata/processed_dataset_manifest.json.
* No data augmentation.  No ML model is trained or loaded.
"""

import json
import shutil
import sys
from pathlib import Path

# ===========================================================================
# Configuration
# ===========================================================================

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
METADATA_DIR   = PROJECT_ROOT / "data" / "metadata"
SPLIT_FILE     = METADATA_DIR / "dataset_split.json"
MANIFEST_FILE  = METADATA_DIR / "processed_dataset_manifest.json"

# Source: flat per-category folders from Phase 5.2
PROCESSED_IMAGES_ROOT = PROJECT_ROOT / "data" / "processed" / "images"

# Destination: organised split tree
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dataset"

# Split folder names (val split recorded as "val" in split.json)
SPLIT_DIR_MAP = {
    "train": "train",
    "val":   "validation",
    "test":  "test",
}

# Expected counts (used for validation)
CATEGORIES = ["invoices", "forms", "resumes", "other"]
EXPECTED_SPLIT_TOTALS = {"train": 1400, "val": 300, "test": 300}
EXPECTED_SPLIT_PER_CAT = {"train": 350, "val": 75, "test": 75}

# Label mapping (kept for manifest; not changed here)
LABEL_MAP = {"invoices": 0, "forms": 1, "resumes": 2, "other": 3}
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ===========================================================================
# Helpers
# ===========================================================================

def load_split():
    """Load dataset_split.json; exit clearly if missing or malformed."""
    if not SPLIT_FILE.exists():
        print(
            f"[ERROR] Split file not found: {SPLIT_FILE}\n"
            "        Run Phase 5.4 (split_dataset.py) first."
        )
        sys.exit(1)

    try:
        with open(SPLIT_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Cannot parse {SPLIT_FILE}: {exc}")
        sys.exit(1)

    if not isinstance(data, dict):
        print("[ERROR] dataset_split.json must contain a JSON object.")
        sys.exit(1)

    images = data.get("images", [])
    if not isinstance(images, list) or not images:
        print("[ERROR] dataset_split.json contains no image records.")
        sys.exit(1)

    return images


def create_directories():
    """Pre-create all 12 leaf directories in the dataset tree."""
    for split_key, split_dir in SPLIT_DIR_MAP.items():
        for cat in CATEGORIES:
            (DATASET_ROOT / split_dir / cat).mkdir(parents=True, exist_ok=True)


def validate_records(records):
    """Validate split records before creating or changing destinations."""
    issues = []
    seen = set()

    for index, rec in enumerate(records, 1):
        if not isinstance(rec, dict):
            issues.append(f"[RECORD] Entry {index} is not an object.")
            continue

        split = rec.get("split")
        category = rec.get("category")
        filename = rec.get("filename")
        label = rec.get("label")

        if split not in SPLIT_DIR_MAP:
            issues.append(f"[SPLIT] Entry {index} has unknown split: {split!r}.")
        if category not in LABEL_MAP:
            issues.append(f"[CATEGORY] Entry {index} has unknown category: {category!r}.")
        if not isinstance(filename, str) or not filename:
            issues.append(f"[FILENAME] Entry {index} has no valid filename.")
        elif Path(filename).name != filename:
            issues.append(f"[FILENAME] Entry {index} contains a path: {filename!r}.")
        elif Path(filename).suffix.lower() not in VALID_EXTENSIONS:
            issues.append(f"[FILENAME] Entry {index} is not a supported image: {filename!r}.")

        if category in LABEL_MAP and label != LABEL_MAP[category]:
            issues.append(
                f"[LABEL] Entry {index} has label {label!r}; "
                f"expected {LABEL_MAP[category]} for {category}."
            )

        if split in SPLIT_DIR_MAP and category in LABEL_MAP and isinstance(filename, str):
            key = (split, category, filename)
            if key in seen:
                issues.append(
                    f"[DUPLICATE] {category}/{filename} appears more than once "
                    f"in {split}."
                )
            seen.add(key)

    return issues


def remove_stale_destinations(records):
    """Remove old managed image files no longer present in the current split."""
    expected = {
        DATASET_ROOT / SPLIT_DIR_MAP[rec["split"]] / rec["category"] / rec["filename"]
        for rec in records
    }
    removed = 0

    if not DATASET_ROOT.exists():
        return removed

    for path in DATASET_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS and path not in expected:
            path.unlink()
            removed += 1

    return removed


def sizes_match(src: Path, dst: Path) -> bool:
    """Return True if dst exists and has the same byte size as src."""
    return dst.exists() and dst.stat().st_size == src.stat().st_size


# ===========================================================================
# Core organisation function
# ===========================================================================

def organize(records):
    """
    Copy each image to its final split/category destination.

    Returns
    -------
    manifest_entries : list[dict]
    stats            : dict  – counters for the summary
    issues           : list[str]
    """
    manifest_entries = []
    issues = []

    # Counters
    copied  = 0
    skipped = 0
    failed  = 0
    bytes_copied = 0

    # Per-split / per-category counter for validation
    dist = {
        sp: {cat: 0 for cat in CATEGORIES}
        for sp in SPLIT_DIR_MAP
    }

    total = len(records)
    print(f"    Processing {total} records …")

    for idx, rec in enumerate(records, 1):
        split_key = rec.get("split")          # "train" | "val" | "test"
        category  = rec.get("category")
        filename  = rec.get("filename")
        label     = rec.get("label", LABEL_MAP.get(category))

        # Resolve source path: data/processed/images/<category>/<filename>
        src_path = PROCESSED_IMAGES_ROOT / category / filename

        # Resolve destination path
        split_dir = SPLIT_DIR_MAP.get(split_key, split_key)
        dst_path  = DATASET_ROOT / split_dir / category / filename

        # ---- Validate source existence --------------------------------
        if not src_path.exists():
            issues.append(f"[MISSING SRC] {src_path.relative_to(PROJECT_ROOT)}")
            failed += 1
            manifest_entries.append({
                "filename":    filename,
                "source_path": str(src_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "final_path":  str(dst_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "category":    category,
                "label":       label,
                "split":       split_key,
                "status":      "failed_missing_source",
            })
            continue

        # ---- Idempotency: skip if destination already up-to-date ------
        if sizes_match(src_path, dst_path):
            skipped += 1
            status = "skipped"
        else:
            # ---- Copy the file ----------------------------------------
            try:
                shutil.copy2(src_path, dst_path)
                file_size = dst_path.stat().st_size
                bytes_copied += file_size
                copied += 1
                status = "copied"
            except OSError as exc:
                issues.append(
                    f"[COPY ERROR] {filename}: {exc}"
                )
                failed += 1
                status = "failed_copy_error"

        # Update distribution counter only for successful outcomes
        if status in ("copied", "skipped"):
            dist[split_key][category] += 1

        manifest_entries.append({
            "filename":    filename,
            "source_path": str(src_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "final_path":  str(dst_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "category":    category,
            "label":       label,
            "split":       split_key,
            "status":      status,
        })

        # Progress indicator every 200 files
        if idx % 200 == 0 or idx == total:
            print(f"      {idx}/{total} — copied: {copied}  "
                  f"skipped: {skipped}  failed: {failed}")

    stats = {
        "copied":       copied,
        "skipped":      skipped,
        "failed":       failed,
        "bytes_copied": bytes_copied,
        "distribution": dist,
    }
    return manifest_entries, stats, issues


# ===========================================================================
# Validation
# ===========================================================================

def validate(stats, issues):
    """Compare actual distribution against expected counts."""
    dist = stats["distribution"]

    for sp, expected_total in EXPECTED_SPLIT_TOTALS.items():
        actual_total = sum(dist[sp].values())
        if actual_total != expected_total:
            issues.append(
                f"[COUNT] {sp}: expected {expected_total}, got {actual_total}."
            )

    expected_per_cat = EXPECTED_SPLIT_PER_CAT
    for sp in SPLIT_DIR_MAP:
        for cat in CATEGORIES:
            actual   = dist[sp][cat]
            expected = expected_per_cat[sp]
            if actual != expected:
                issues.append(
                    f"[COUNT] {sp}/{cat}: expected {expected}, got {actual}."
                )


# ===========================================================================
# Save manifest
# ===========================================================================

def save_manifest(manifest_entries, stats, issues):
    """Write processed_dataset_manifest.json."""
    dist     = stats["distribution"]
    split_totals = {sp: sum(dist[sp].values()) for sp in SPLIT_DIR_MAP}

    payload = {
        "summary": {
            "total_images":  len(manifest_entries),
            "copied":        stats["copied"],
            "skipped":       stats["skipped"],
            "failed":        stats["failed"],
            "bytes_copied":  stats["bytes_copied"],
            "split_totals":  split_totals,
            "distribution":  dist,
        },
        "issues":  issues,
        "images":  manifest_entries,
    }

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("SmartDocs AI - Phase 5.6: Dataset Organization")
    print("=" * 60)

    # ---- Step 1: Load split info ----------------------------------------
    print("\nStep 1: Loading dataset_split.json ...")
    records = load_split()
    print(f"  Loaded {len(records)} record(s).")

    print("\nStep 2: Validating split records ...")
    record_issues = validate_records(records)
    if record_issues:
        for issue in record_issues:
            print(f"  {issue}")
        print("  No destination files were changed.")
        sys.exit(1)
    print("  Split records are valid and unique.")

    # ---- Step 3: Create directory tree ----------------------------------
    print("\nStep 3: Creating directory structure ...")
    create_directories()
    print("  All 12 leaf directories ready.")

    removed = remove_stale_destinations(records)
    if removed:
        print(f"  Removed {removed} stale managed image file(s).")

    # ---- Step 4: Copy / organise images ---------------------------------
    print("\nStep 4: Organising images ...")
    manifest_entries, stats, issues = organize(records)

    # ---- Step 5: Validate -----------------------------------------------
    print("\nStep 5: Validating counts ...")
    validate(stats, issues)

    if issues:
        print(f"  {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  All counts verified. No issues.")

    # ---- Step 6: Save manifest ------------------------------------------
    print("\nStep 6: Writing manifest ...")
    save_manifest(manifest_entries, stats, issues)
    print(f"  Saved -> {MANIFEST_FILE}")

    # ---- Final summary --------------------------------------------------
    dist = stats["distribution"]
    split_totals = {sp: sum(dist[sp].values()) for sp in SPLIT_DIR_MAP}
    mb_copied = stats["bytes_copied"] / (1024 * 1024)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total records processed  : {len(manifest_entries)}")
    print(f"  Files copied             : {stats['copied']}")
    print(f"  Files skipped (re-run)   : {stats['skipped']}")
    print(f"  Failed                   : {stats['failed']}")
    print(f"  Additional disk usage    : {mb_copied:.1f} MB (copies)")
    print()

    # Distribution table
    col_w = 12
    header = f"  {'Category':<10}" + "".join(
        f"  {SPLIT_DIR_MAP[sp].capitalize():>{col_w}}"
        for sp in ("train", "val", "test")
    )
    print(header)
    print("  " + "-" * (10 + 3 * (col_w + 2)))
    for cat in CATEGORIES:
        row = f"  {cat:<10}" + "".join(
            f"  {dist[sp][cat]:>{col_w}}" for sp in ("train", "val", "test")
        )
        print(row)
    print("  " + "-" * (10 + 3 * (col_w + 2)))
    total_row = f"  {'TOTAL':<10}" + "".join(
        f"  {split_totals[sp]:>{col_w}}" for sp in ("train", "val", "test")
    )
    print(total_row)
    print()

    print(f"  Final dataset location   : {DATASET_ROOT}")
    print()

    if issues:
        print(f"  Issues / warnings : {len(issues)}")
        for issue in issues:
            print(f"    {issue}")
        print("\n  Organization result: INCOMPLETE")
    else:
        print("  Issues / warnings : None")
        print("\n  Organization result: COMPLETE")

    print("=" * 60)
    print("Phase 5.6 complete.\n")

    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
