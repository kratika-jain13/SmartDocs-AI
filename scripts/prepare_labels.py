"""
Phase 5.3 – Label Preparation
==============================
SmartDocs AI Project

Purpose
-------
Walk through data/processed/images/<category>/ for all four categories,
assign each image a fixed integer label, and write the complete label
dataset to data/metadata/labels.json.

Fixed label mapping
-------------------
    invoices  -> 0
    forms     -> 1
    resumes   -> 2
    other     -> 3

Rules
-----
* data/raw/ is never touched.
* Processed images are never modified.
* No train/val/test splitting is done here.
* No data augmentation.
* No ML model is trained or loaded.
* Running the script multiple times is safe (labels.json is overwritten
  with a fresh, consistent result every time).
"""

import json
import sys
from pathlib import Path

# ===========================================================================
# Configuration
# ===========================================================================

# Project root: two levels up from scripts/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Root of the processed images
PROCESSED_IMAGES_ROOT = PROJECT_ROOT / "data" / "processed" / "images"

# Output metadata directory and file
METADATA_DIR  = PROJECT_ROOT / "data" / "metadata"
LABELS_FILE   = METADATA_DIR / "labels.json"

# Fixed label mapping – must not be changed without retraining downstream models
LABEL_MAP = {
    "invoices": 0,
    "forms":    1,
    "resumes":  2,
    "other":    3,
}

# Expected number of images per category (used for validation)
EXPECTED_PER_CATEGORY = 500

# Image file extensions to recognise
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ===========================================================================
# Helper utilities
# ===========================================================================

def ensure_directories():
    """Create output directories if they do not already exist."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def collect_labeled_records():
    """
    Walk every category directory under data/processed/images/ and build a
    list of label records – one per image file found.

    Returns
    -------
    records : list[dict]
        Each dict has keys: filename, relative_path, category, label
    issues  : list[str]
        Human-readable descriptions of any problems found during collection
    """
    records = []
    issues  = []

    for category, label in LABEL_MAP.items():
        category_dir = PROCESSED_IMAGES_ROOT / category

        # ---- Check that the source directory actually exists ----
        if not category_dir.exists():
            issues.append(
                f"[MISSING DIR] data/processed/images/{category}/ does not exist."
            )
            continue

        # ---- Collect image files (sorted for deterministic ordering) ----
        image_files = sorted(
            p for p in category_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )

        if not image_files:
            issues.append(
                f"[EMPTY DIR]  data/processed/images/{category}/ contains no images."
            )

        for image_path in image_files:
            # relative_path is relative to data/processed/ so callers always
            # know exactly where to find the file without embedding an
            # absolute machine-specific path.
            relative_path = f"images/{category}/{image_path.name}"

            records.append(
                {
                    "filename":      image_path.name,
                    "relative_path": relative_path,
                    "category":      category,
                    "label":         label,
                }
            )

    return records, issues


def validate_records(records, issues):
    """
    Run post-collection validation checks and append any findings to `issues`.

    Checks performed
    ----------------
    1. Every processed image has exactly one label (no duplicates).
    2. Each category has exactly EXPECTED_PER_CATEGORY images.
    3. Every label value is a known integer from LABEL_MAP.

    Parameters
    ----------
    records : list[dict]  – output of collect_labeled_records()
    issues  : list[str]   – existing issue list (mutated in-place)
    """
    # --- Duplicate filename check (within the same category) ---------------
    seen_keys = {}
    for rec in records:
        key = f"{rec['category']}/{rec['filename']}"
        if key in seen_keys:
            issues.append(f"[DUPLICATE]  {key} appears more than once.")
        else:
            seen_keys[key] = True

    # --- Per-category count check ------------------------------------------
    category_counts = {cat: 0 for cat in LABEL_MAP}
    for rec in records:
        cat = rec["category"]
        if cat in category_counts:
            category_counts[cat] += 1
        else:
            issues.append(
                f"[UNKNOWN CAT] Record has unrecognised category: {cat!r}"
            )

    for cat, count in category_counts.items():
        if count != EXPECTED_PER_CATEGORY:
            issues.append(
                f"[COUNT MISMATCH] {cat}: expected {EXPECTED_PER_CATEGORY}, "
                f"found {count}."
            )

    # --- Label integrity check ---------------------------------------------
    valid_labels = set(LABEL_MAP.values())
    for rec in records:
        if rec["label"] not in valid_labels:
            issues.append(
                f"[BAD LABEL]  {rec['filename']} has unknown label {rec['label']!r}."
            )

    return category_counts


def save_labels(records, category_counts, issues):
    """
    Write the label dataset to data/metadata/labels.json.

    The file contains:
      - label_map      : the fixed integer mapping used
      - summary        : total count and per-category counts
      - issues         : any validation warnings recorded
      - images         : the full list of per-image records
    """
    payload = {
        "label_map": LABEL_MAP,
        "summary": {
            "total_labeled_images": len(records),
            "per_category":         category_counts,
        },
        "issues":  issues,
        "images":  records,
    }

    with open(LABELS_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


# ===========================================================================
# Main orchestration
# ===========================================================================

def main():
    print("=" * 60)
    print("SmartDocs AI - Phase 5.3: Label Preparation")
    print("=" * 60)

    # ---- Setup -----------------------------------------------------------
    ensure_directories()

    # ---- Step 1: Collect records -----------------------------------------
    print("\nStep 1: Collecting image files from processed dataset...")
    records, issues = collect_labeled_records()
    print(f"  Collected {len(records)} image record(s).")

    # ---- Step 2: Validate ------------------------------------------------
    print("\nStep 2: Validating label integrity...")
    category_counts = validate_records(records, issues)

    if issues:
        print(f"  {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  All validation checks passed.")

    # ---- Step 3: Save labels.json ----------------------------------------
    print("\nStep 3: Writing labels.json...")
    save_labels(records, category_counts, issues)
    print(f"  Saved -> {LABELS_FILE}")

    # ---- Final summary ---------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total labeled images : {len(records)}")
    print()
    print("  Label mapping:")
    for cat, lbl in LABEL_MAP.items():
        print(f"    {cat:<10} -> {lbl}")
    print()
    print("  Count per category:")
    for cat, count in category_counts.items():
        status = "OK" if count == EXPECTED_PER_CATEGORY else "MISMATCH"
        print(f"    {cat:<10} : {count:>4}  [{status}]")
    print()
    if issues:
        print(f"  Issues / warnings : {len(issues)}")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  Issues / warnings : None")
    print("=" * 60)
    print("Phase 5.3 complete.\n")

    # Exit with non-zero code if any issues were found so CI pipelines can
    # detect problems automatically.
    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
