"""
Phase 5.4 – Train / Validation / Test Split
============================================
SmartDocs AI Project

Purpose
-------
Read the label dataset from data/metadata/labels.json, perform a
stratified 70 / 15 / 15 split so every category stays balanced, and
write the complete split record to data/metadata/dataset_split.json.

Split ratios
------------
    Training   : 70%  -> 1 400 images  (350 per category)
    Validation : 15%  ->   300 images  ( 75 per category)
    Test       : 15%  ->   300 images  ( 75 per category)

Rules
-----
* data/raw/ is never touched.
* Processed images are never modified.
* No train/val/test data is physically moved – only the JSON metadata
  records which split each image belongs to.
* No data augmentation.
* No ML model is trained or loaded.
* Fixed random seed ensures the split is reproducible across runs.
* Safe to run multiple times – dataset_split.json is overwritten each run.
"""

import json
import random
import sys
from pathlib import Path

# ===========================================================================
# Configuration
# ===========================================================================

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
METADATA_DIR  = PROJECT_ROOT / "data" / "metadata"
LABELS_FILE   = METADATA_DIR / "labels.json"
SPLIT_FILE    = METADATA_DIR / "dataset_split.json"

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Fixed seed for full reproducibility
RANDOM_SEED = 42

# Expected counts (used for validation)
TOTAL_EXPECTED          = 2000
EXPECTED_PER_CATEGORY   = 500
CATEGORIES              = ["invoices", "forms", "resumes", "other"]

EXPECTED_SPLIT_COUNTS = {
    "train": 1400,
    "val":    300,
    "test":   300,
}
EXPECTED_SPLIT_PER_CAT = {
    "train": 350,
    "val":    75,
    "test":   75,
}


# ===========================================================================
# Helper utilities
# ===========================================================================

def load_labels():
    """
    Load data/metadata/labels.json and return the list of image records.
    Exits with an error message if the file is missing or malformed.
    """
    if not LABELS_FILE.exists():
        print(
            f"[ERROR] Labels file not found: {LABELS_FILE}\n"
            "        Run Phase 5.3 (prepare_labels.py) first."
        )
        sys.exit(1)

    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Could not parse {LABELS_FILE}: {exc}")
        sys.exit(1)

    images = data.get("images", [])
    if not images:
        print("[ERROR] labels.json contains no image records.")
        sys.exit(1)

    return images


def stratified_split(records, train_ratio, val_ratio, seed):
    """
    Perform a stratified split of `records` by the 'category' field.

    For each category the images are shuffled with `seed` and then divided
    into train / val / test slices using integer counts derived from
    `train_ratio` and `val_ratio` (test gets the remainder so the totals
    are exact with no rounding loss).

    Parameters
    ----------
    records     : list[dict]  – full label records from labels.json
    train_ratio : float       – fraction for training  (e.g. 0.70)
    val_ratio   : float       – fraction for validation (e.g. 0.15)
    seed        : int         – random seed for reproducibility

    Returns
    -------
    split_records : list[dict]
        A copy of every record with an added 'split' key
        ("train" | "val" | "test").
    """
    rng = random.Random(seed)

    # Group records by category
    by_category = {cat: [] for cat in CATEGORIES}
    for rec in records:
        cat = rec.get("category")
        if cat in by_category:
            by_category[cat].append(rec)
        else:
            # Unknown category – put in a holding list (validation will flag it)
            by_category.setdefault("__unknown__", []).append(rec)

    split_records = []

    for cat, cat_records in by_category.items():
        # Shuffle deterministically within this category
        shuffled = list(cat_records)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = round(n * train_ratio)
        n_val   = round(n * val_ratio)
        # Test gets the remainder – guarantees n_train + n_val + n_test == n
        n_test  = n - n_train - n_val

        train_slice = shuffled[:n_train]
        val_slice   = shuffled[n_train : n_train + n_val]
        test_slice  = shuffled[n_train + n_val :]

        for rec in train_slice:
            split_records.append({**rec, "split": "train"})
        for rec in val_slice:
            split_records.append({**rec, "split": "val"})
        for rec in test_slice:
            split_records.append({**rec, "split": "test"})

    return split_records


def validate_split(split_records):
    """
    Run integrity checks on the split result.

    Checks
    ------
    1. Total record count == TOTAL_EXPECTED
    2. Every image appears exactly once (no duplicates, no missing)
    3. Per-split totals match EXPECTED_SPLIT_COUNTS
    4. Per-split per-category counts match EXPECTED_SPLIT_PER_CAT

    Returns
    -------
    issues     : list[str]  – human-readable problem descriptions
    stats      : dict       – aggregated counts for the summary printout
    """
    issues = []

    # ---- Total count -------------------------------------------------------
    total = len(split_records)
    if total != TOTAL_EXPECTED:
        issues.append(
            f"[COUNT] Total records: expected {TOTAL_EXPECTED}, got {total}."
        )

    # ---- Duplicate / uniqueness check -------------------------------------
    seen_keys = {}
    for rec in split_records:
        key = f"{rec['category']}/{rec['filename']}"
        if key in seen_keys:
            issues.append(f"[DUPLICATE] {key} appears more than once.")
        else:
            seen_keys[key] = True

    # ---- Per-split stats --------------------------------------------------
    stats = {
        split: {cat: 0 for cat in CATEGORIES}
        for split in ("train", "val", "test")
    }
    split_totals = {"train": 0, "val": 0, "test": 0}

    for rec in split_records:
        sp  = rec.get("split")
        cat = rec.get("category")
        if sp in split_totals:
            split_totals[sp] += 1
        if sp in stats and cat in CATEGORIES:
            stats[sp][cat] += 1

    # ---- Validate per-split totals ----------------------------------------
    for sp, expected in EXPECTED_SPLIT_COUNTS.items():
        actual = split_totals[sp]
        if actual != expected:
            issues.append(
                f"[SPLIT COUNT] {sp}: expected {expected}, got {actual}."
            )

    # ---- Validate per-split per-category counts ---------------------------
    for sp in ("train", "val", "test"):
        for cat in CATEGORIES:
            actual   = stats[sp][cat]
            expected = EXPECTED_SPLIT_PER_CAT[sp]
            if actual != expected:
                issues.append(
                    f"[CAT COUNT] {sp}/{cat}: expected {expected}, got {actual}."
                )

    return issues, stats, split_totals


def save_split(split_records, stats, split_totals, issues):
    """
    Write the split dataset to data/metadata/dataset_split.json.
    """
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val":   VAL_RATIO,
            "test":  TEST_RATIO,
        },
        "summary": {
            "total":       len(split_records),
            "train":       split_totals["train"],
            "val":         split_totals["val"],
            "test":        split_totals["test"],
            "distribution": stats,
        },
        "issues":  issues,
        "images":  split_records,
    }

    with open(SPLIT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


# ===========================================================================
# Main orchestration
# ===========================================================================

def main():
    print("=" * 60)
    print("SmartDocs AI - Phase 5.4: Train / Validation / Test Split")
    print(f"Seed: {RANDOM_SEED}  |  Ratios: "
          f"train={TRAIN_RATIO}  val={VAL_RATIO}  test={TEST_RATIO}")
    print("=" * 60)

    # ---- Step 1: Load labels ---------------------------------------------
    print("\nStep 1: Loading label dataset...")
    records = load_labels()
    print(f"  Loaded {len(records)} labeled image record(s).")

    # ---- Step 2: Stratified split ----------------------------------------
    print("\nStep 2: Performing stratified split...")
    split_records = stratified_split(records, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    print(f"  Split complete — {len(split_records)} records assigned.")

    # ---- Step 3: Validate ------------------------------------------------
    print("\nStep 3: Validating split integrity...")
    issues, stats, split_totals = validate_split(split_records)

    if issues:
        print(f"  {len(issues)} issue(s) detected:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  All validation checks passed.")

    # ---- Step 4: Save dataset_split.json ---------------------------------
    print("\nStep 4: Writing dataset_split.json...")
    save_split(split_records, stats, split_totals, issues)
    print(f"  Saved -> {SPLIT_FILE}")

    # ---- Final summary ---------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total images         : {len(split_records)}")
    print(f"  Training  (70%)      : {split_totals['train']}")
    print(f"  Validation (15%)     : {split_totals['val']}")
    print(f"  Test       (15%)     : {split_totals['test']}")
    print()

    # Category distribution table
    col_w = 12
    header = f"  {'Category':<10}" + "".join(
        f"  {sp.capitalize():>{col_w}}" for sp in ("train", "val", "test")
    )
    print(header)
    print("  " + "-" * (10 + 3 * (col_w + 2)))
    for cat in CATEGORIES:
        row = f"  {cat:<10}" + "".join(
            f"  {stats[sp][cat]:>{col_w}}" for sp in ("train", "val", "test")
        )
        print(row)
    print("  " + "-" * (10 + 3 * (col_w + 2)))
    totals_row = f"  {'TOTAL':<10}" + "".join(
        f"  {split_totals[sp]:>{col_w}}" for sp in ("train", "val", "test")
    )
    print(totals_row)
    print()

    if issues:
        print(f"  Validation issues    : {len(issues)}")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  Validation issues    : None")

    print("=" * 60)
    print("Phase 5.4 complete.\n")

    # Non-zero exit code lets CI pipelines detect split integrity failures
    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
