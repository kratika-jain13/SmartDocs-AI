"""
Phase 5.5 – Data Leakage Prevention
=====================================
SmartDocs AI Project

Purpose
-------
Load the split metadata from data/metadata/dataset_split.json and perform
two complementary leakage checks:

  1. Filename-based overlap check
     Verify that no image filename appears in more than one split (train,
     val, test).

  2. Content-based duplicate check (SHA-256 hash)
     Compute a cryptographic hash of every processed image file and verify
     that no two images from different splits share identical content,
     even if they carry different filenames.

A report is written to data/metadata/data_leakage_report.json and a
clear PASS / FAIL verdict is printed to the terminal.

Rules
-----
* Images are only READ – never moved, renamed, or deleted.
* data/raw/ is never touched.
* No ML model is trained or loaded.
* No data augmentation.
* Safe to run multiple times – the report is overwritten each run.
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

# ===========================================================================
# Configuration
# ===========================================================================

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
METADATA_DIR  = PROJECT_ROOT / "data" / "metadata"
SPLIT_FILE    = METADATA_DIR / "dataset_split.json"
REPORT_FILE   = METADATA_DIR / "data_leakage_report.json"

# Root that relative_path values inside dataset_split.json are relative to
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

# Expected split sizes
EXPECTED_COUNTS = {"train": 1400, "val": 300, "test": 300}
EXPECTED_TOTAL  = 2000

# SHA-256 hash chunk size (64 KiB) – balances speed vs memory
HASH_CHUNK = 65536


# ===========================================================================
# Helpers
# ===========================================================================

def load_split():
    """Load dataset_split.json; exit with a clear message if missing."""
    if not SPLIT_FILE.exists():
        print(
            f"[ERROR] Split file not found: {SPLIT_FILE}\n"
            "        Run Phase 5.4 (split_dataset.py) first."
        )
        sys.exit(1)

    with open(SPLIT_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    images = data.get("images", [])
    if not images:
        print("[ERROR] dataset_split.json contains no image records.")
        sys.exit(1)

    return images


def sha256_of_file(file_path: Path) -> str | None:
    """
    Return the SHA-256 hex digest of a file's contents, or None if the file
    cannot be read (the issue is reported elsewhere).
    """
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as fh:
            while chunk := fh.read(HASH_CHUNK):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        return None


# ===========================================================================
# Check 1 – Filename-based overlap
# ===========================================================================

def check_filename_overlap(records):
    """
    Build a mapping of  canonical_key -> [splits it appears in]  and flag
    any key that appears in more than one split.

    canonical_key = "<category>/<filename>"  (mirrors dataset_split.json)

    Returns
    -------
    overlaps : list[dict]   – one entry per duplicated key
    split_sets : dict       – sets of keys per split, for reuse
    """
    # Map each canonical key to the list of splits it is assigned to
    key_to_splits: dict[str, list[str]] = defaultdict(list)

    for rec in records:
        key = f"{rec['category']}/{rec['filename']}"
        key_to_splits[key].append(rec["split"])

    overlaps = []
    for key, splits in key_to_splits.items():
        if len(splits) > 1:
            overlaps.append({"image": key, "found_in_splits": splits})

    # Build per-split key sets (used by the summary and content check)
    split_sets: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for rec in records:
        key = f"{rec['category']}/{rec['filename']}"
        split_sets[rec["split"]].add(key)

    return overlaps, split_sets


# ===========================================================================
# Check 2 – Content-based duplicate (SHA-256)
# ===========================================================================

def check_content_duplicates(records):
    """
    Compute SHA-256 for every processed image and detect cases where the
    same file content appears in more than one split.

    Returns
    -------
    content_duplicates : list[dict]
        Each entry describes a hash collision across splits.
    missing_files      : list[str]
        Relative paths of files that could not be found / read.
    hash_map           : dict[str, list[dict]]
        Full hash -> [record, …] mapping (for the report).
    """
    # hash -> list of {split, relative_path, category, filename}
    hash_map: dict[str, list[dict]] = defaultdict(list)
    missing_files: list[str] = []

    total = len(records)
    print(f"    Hashing {total} image files (SHA-256) — this may take a moment…")

    for idx, rec in enumerate(records, 1):
        rel_path   = rec.get("relative_path", "")
        file_path  = PROCESSED_ROOT / rel_path

        if not file_path.exists():
            missing_files.append(rel_path)
            continue

        digest = sha256_of_file(file_path)
        if digest is None:
            missing_files.append(rel_path)
            continue

        hash_map[digest].append(
            {
                "split":         rec["split"],
                "relative_path": rel_path,
                "category":      rec["category"],
                "filename":      rec["filename"],
            }
        )

        # Progress indicator every 200 files
        if idx % 200 == 0 or idx == total:
            print(f"      Hashed {idx}/{total} …")

    # Find hashes whose images span more than one split
    content_duplicates = []
    for digest, entries in hash_map.items():
        splits_present = {e["split"] for e in entries}
        if len(splits_present) > 1:
            content_duplicates.append(
                {
                    "sha256":   digest,
                    "copies":   entries,
                    "in_splits": sorted(splits_present),
                }
            )

    return content_duplicates, missing_files, dict(hash_map)


# ===========================================================================
# Save report
# ===========================================================================

def save_report(records, split_sets, filename_overlaps,
                content_duplicates, missing_files, leakage_detected):
    """Write data/metadata/data_leakage_report.json."""

    split_counts = {sp: len(s) for sp, s in split_sets.items()}

    # Pairwise filename overlaps (derived from split_sets for completeness)
    pairwise = {
        "train_vs_val":  sorted(split_sets["train"] & split_sets["val"]),
        "train_vs_test": sorted(split_sets["train"] & split_sets["test"]),
        "val_vs_test":   sorted(split_sets["val"]   & split_sets["test"]),
    }

    report = {
        "leakage_status": "FAIL" if leakage_detected else "PASS",
        "total_records":  len(records),
        "unique_records": len({r['category'] + '/' + r['filename'] for r in records}),
        "split_counts":   split_counts,
        "expected_counts": EXPECTED_COUNTS,
        "count_check": {
            sp: ("OK" if split_counts.get(sp, 0) == exp else "MISMATCH")
            for sp, exp in EXPECTED_COUNTS.items()
        },
        "filename_overlap": {
            "pairwise_overlaps":   pairwise,
            "duplicate_records":   filename_overlaps,
            "any_overlap_detected": bool(filename_overlaps),
        },
        "content_duplicate_check": {
            "missing_files":           missing_files,
            "cross_split_duplicates":  content_duplicates,
            "any_duplicate_detected":  bool(content_duplicates),
        },
    }

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("SmartDocs AI - Phase 5.5: Data Leakage Prevention")
    print("=" * 60)

    # ---- Load split metadata ---------------------------------------------
    print("\nStep 1: Loading dataset_split.json …")
    records = load_split()
    print(f"  Loaded {len(records)} record(s).")

    # ---- Count validation ------------------------------------------------
    print("\nStep 2: Verifying split counts …")
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for rec in records:
        sp = rec.get("split")
        if sp in split_counts:
            split_counts[sp] += 1

    count_issues = []
    for sp, expected in EXPECTED_COUNTS.items():
        actual = split_counts[sp]
        status = "OK" if actual == expected else f"MISMATCH (expected {expected})"
        print(f"    {sp:<6}: {actual:>5}  [{status}]")
        if actual != expected:
            count_issues.append(f"{sp}: expected {expected}, got {actual}")

    total_actual = sum(split_counts.values())
    total_status = "OK" if total_actual == EXPECTED_TOTAL else "MISMATCH"
    print(f"    {'TOTAL':<6}: {total_actual:>5}  [{total_status}]")

    # ---- Filename overlap check ------------------------------------------
    print("\nStep 3: Filename-based overlap check …")
    filename_overlaps, split_sets = check_filename_overlap(records)

    pairwise_counts = {
        "train and val":  len(split_sets["train"] & split_sets["val"]),
        "train and test": len(split_sets["train"] & split_sets["test"]),
        "val   and test": len(split_sets["val"]   & split_sets["test"]),
    }
    for pair, count in pairwise_counts.items():
        status = "CLEAN" if count == 0 else f"OVERLAP ({count} image(s))"
        print(f"    {pair}: {status}")

    if filename_overlaps:
        print(f"  !! {len(filename_overlaps)} filename overlap(s) detected.")
    else:
        print("  Filename check: no overlaps found.")

    # ---- Content duplicate check ----------------------------------------
    print("\nStep 4: Content-based duplicate check (SHA-256) …")
    content_duplicates, missing_files, _ = check_content_duplicates(records)

    if missing_files:
        print(f"  !! {len(missing_files)} file(s) could not be read:")
        for mf in missing_files[:10]:          # show at most 10
            print(f"       {mf}")
        if len(missing_files) > 10:
            print(f"       … and {len(missing_files) - 10} more.")
    else:
        print("  All files readable.")

    if content_duplicates:
        print(f"  !! {len(content_duplicates)} cross-split content duplicate(s) found.")
        for dup in content_duplicates[:5]:
            print(f"     hash={dup['sha256'][:16]}…  in splits: {dup['in_splits']}")
    else:
        print("  Content check: no cross-split duplicates found.")

    # ---- Determine overall result ----------------------------------------
    leakage_detected = bool(
        filename_overlaps or content_duplicates or missing_files or count_issues
    )

    # ---- Save report -----------------------------------------------------
    print("\nStep 5: Writing data_leakage_report.json …")
    save_report(
        records, split_sets, filename_overlaps,
        content_duplicates, missing_files, leakage_detected
    )
    print(f"  Saved -> {REPORT_FILE}")

    # ---- Final summary ---------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    unique_count = len({r['category'] + '/' + r['filename'] for r in records})
    print(f"  Total records            : {len(records)}")
    print(f"  Unique images            : {unique_count}")
    print(f"  Train                    : {split_counts['train']}")
    print(f"  Validation               : {split_counts['val']}")
    print(f"  Test                     : {split_counts['test']}")
    print(f"  Filename overlaps        : {len(filename_overlaps)}")
    print(f"  Cross-split content dups : {len(content_duplicates)}")
    print(f"  Missing / unreadable     : {len(missing_files)}")
    print(f"  Count mismatches         : {len(count_issues)}")
    print()

    verdict = "FAIL" if leakage_detected else "PASS"
    border  = "!!" if leakage_detected else "**"
    print(f"  {border} Data Leakage Check: {verdict} {border}")
    print("=" * 60)
    print("Phase 5.5 complete.\n")

    if leakage_detected:
        sys.exit(1)


if __name__ == "__main__":
    main()
