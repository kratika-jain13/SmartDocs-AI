"""
Phase 5.2 – Image Preprocessing
================================
SmartDocs AI Project

Purpose
-------
Read raw document images (invoices, forms, resumes, other), validate that
each file can be opened, convert to RGB colour mode, resize to 224×224 px,
and save the results under data/processed/images/<category>/.

A JSON metadata file is written to data/metadata/preprocessing_metadata.json
that records per-image details and the overall run summary.

Rules
-----
* data/raw/ is NEVER modified or deleted.
* No data augmentation is performed.
* No ML model is trained or loaded.
* Running the script multiple times is safe – already-processed images are
  skipped and their existing metadata entry is reused.
* Corrupted / unreadable images are skipped gracefully; they are logged as
  "failed" in the metadata without crashing the program.
"""

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pillow is the only non-stdlib dependency required for image I/O and resize.
# ---------------------------------------------------------------------------
try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print(
        "[ERROR] Pillow is not installed.\n"
        "        Run:  pip install Pillow\n"
        "        then re-run this script."
    )
    sys.exit(1)

# ===========================================================================
# Configuration
# ===========================================================================

# Project root is two levels up from this script (scripts/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Categories to process  (folder name == label)
CATEGORIES = ["invoices", "forms", "resumes", "other"]

# Source and destination directory roots
RAW_ROOT       = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "images"
METADATA_DIR   = PROJECT_ROOT / "data" / "metadata"
METADATA_FILE  = METADATA_DIR / "preprocessing_metadata.json"

# Target size for every output image
TARGET_SIZE = (224, 224)  # (width, height) in pixels

# Image file extensions to consider
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ===========================================================================
# Helper utilities
# ===========================================================================

def ensure_directories():
    """Create all required output directories if they do not already exist."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        (PROCESSED_ROOT / category).mkdir(parents=True, exist_ok=True)


def load_existing_metadata():
    """
    Load the metadata file that may have been written by a previous run.

    Returns a dict keyed by "<category>/<filename>" so we can detect
    already-processed images and skip duplicates efficiently.
    """
    if not METADATA_FILE.exists():
        return {}

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Could not read existing metadata ({exc}). Starting fresh.")
        return {}

    # Build a lookup dict from the flat list
    existing = {}
    for entry in data.get("images", []):
        key = f"{entry['category']}/{entry['original_filename']}"
        existing[key] = entry

    return existing


def save_metadata(all_entries, summary):
    """
    Persist the combined metadata (all categories, all images) to disk as JSON.
    """
    payload = {
        "summary": summary,
        "images": all_entries,
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def collect_image_paths(category):
    """
    Return a sorted list of image file Paths found in data/raw/<category>/.
    Only files with a recognised extension are returned.
    Non-image files (e.g. .txt) are silently ignored.
    """
    source_dir = RAW_ROOT / category
    if not source_dir.exists():
        print(f"[WARN] Source directory not found, skipping: {source_dir}")
        return []

    paths = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    return paths


# ===========================================================================
# Core processing function
# ===========================================================================

def process_image(image_path, category, existing_metadata):
    """
    Validate, convert, resize, and save a single image.

    Parameters
    ----------
    image_path        : absolute Path to the raw image file
    category          : category label (e.g. "invoices")
    existing_metadata : dict of already-processed entries from a previous run

    Returns
    -------
    A metadata dict for this image with keys:
      original_filename, category,
      original_width, original_height,
      processed_width, processed_height,
      processing_status  ("success" | "failed" | "skipped")
      error_message       (only present when processing_status == "failed")
    """
    filename  = image_path.name
    lookup_key = f"{category}/{filename}"
    dest_path  = PROCESSED_ROOT / category / filename

    # ------------------------------------------------------------------
    # Idempotency check – if the output file already exists and it was
    # successfully processed before, reuse the old metadata entry and skip.
    # ------------------------------------------------------------------
    if dest_path.exists() and lookup_key in existing_metadata:
        prev = existing_metadata[lookup_key]
        if prev.get("processing_status") == "success":
            # Return the cached entry with status updated to "skipped" for
            # the live counter, but preserve "success" in the saved record.
            cached = dict(prev)
            cached["processing_status"] = "skipped"
            return cached

    # ------------------------------------------------------------------
    # Step 1 – Try to open the image and read its original dimensions.
    # ------------------------------------------------------------------
    try:
        with Image.open(image_path) as img:
            # verify() confirms the file is not truncated / corrupt.
            # It exhausts the file pointer, so we must re-open below.
            img.verify()

        # Re-open after verify() to access pixel data
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            # ----------------------------------------------------------
            # Step 2 – Convert to RGB (handles grayscale, RGBA, palette…)
            # ----------------------------------------------------------
            rgb_img = img.convert("RGB")

            # ----------------------------------------------------------
            # Step 3 – Resize to TARGET_SIZE with high-quality resampling
            # ----------------------------------------------------------
            resized_img = rgb_img.resize(TARGET_SIZE, Image.LANCZOS)

            # ----------------------------------------------------------
            # Step 4 – Save to the processed directory
            # ----------------------------------------------------------
            resized_img.save(dest_path, format="PNG", optimize=True)

    except (UnidentifiedImageError, OSError, SyntaxError, Exception) as exc:
        # Any failure during open / verify / convert / resize / save is caught
        # here.  Record the image as failed and continue with the next file.
        return {
            "original_filename": filename,
            "category":          category,
            "original_width":    None,
            "original_height":   None,
            "processed_width":   None,
            "processed_height":  None,
            "processing_status": "failed",
            "error_message":     str(exc),
        }

    # ------------------------------------------------------------------
    # Step 5 – Build and return the success metadata entry
    # ------------------------------------------------------------------
    processed_width, processed_height = TARGET_SIZE
    return {
        "original_filename": filename,
        "category":          category,
        "original_width":    original_width,
        "original_height":   original_height,
        "processed_width":   processed_width,
        "processed_height":  processed_height,
        "processing_status": "success",
    }


# ===========================================================================
# Main orchestration
# ===========================================================================

def main():
    print("=" * 60)
    print("SmartDocs AI - Phase 5.2: Image Preprocessing")
    print(f"Target size : {TARGET_SIZE[0]}x{TARGET_SIZE[1]} px  |  Mode: RGB")
    print("=" * 60)

    # ---- Setup --------------------------------------------------------
    ensure_directories()
    existing_metadata = load_existing_metadata()

    # ---- Counters -----------------------------------------------------
    total_found   = 0
    total_success = 0
    total_failed  = 0
    total_skipped = 0
    category_counts = {
        cat: {"found": 0, "success": 0, "failed": 0, "skipped": 0}
        for cat in CATEGORIES
    }

    # Collect all metadata entries (existing + new) keyed by category/filename
    # Using a dict lets us overwrite stale "failed" entries on a retry run.
    all_entries_map = {}

    # ---- Process each category ----------------------------------------
    for category in CATEGORIES:
        print(f"\n[Category: {category.upper()}]")

        image_paths = collect_image_paths(category)
        cat_found   = len(image_paths)
        category_counts[category]["found"] = cat_found
        total_found += cat_found

        if cat_found == 0:
            print(f"  No image files found in data/raw/{category}/")
            continue

        print(f"  Found {cat_found} image(s) - processing...")

        for image_path in image_paths:
            result = process_image(image_path, category, existing_metadata)
            status = result["processing_status"]
            key    = f"{category}/{result['original_filename']}"

            # If the entry was skipped (already done), restore "success" status
            # for the saved metadata so the file is still recognised next time.
            saved_result = dict(result)
            if status == "skipped":
                saved_result["processing_status"] = "success"

            # Store / overwrite in the combined map
            all_entries_map[key] = saved_result

            if status == "success":
                category_counts[category]["success"] += 1
                total_success += 1
                print(
                    f"    OK  {result['original_filename']} "
                    f"({result['original_width']}x{result['original_height']} "
                    f"-> {result['processed_width']}x{result['processed_height']})"
                )
            elif status == "skipped":
                category_counts[category]["skipped"] += 1
                total_skipped += 1
                print(f"    --  {result['original_filename']} (already processed, skipped)")
            else:  # failed
                category_counts[category]["failed"] += 1
                total_failed += 1
                print(
                    f"    FAIL  {result['original_filename']} "
                    f"- {result.get('error_message', 'unknown error')}"
                )

    # ---- Build summary dict -------------------------------------------
    summary = {
        "total_images_found":            total_found,
        "total_successfully_processed":  total_success,
        "total_failed":                  total_failed,
        "total_skipped":                 total_skipped,
        "target_size":                   f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
        "colour_mode":                   "RGB",
        "categories": {
            cat: {
                "found":   category_counts[cat]["found"],
                "success": category_counts[cat]["success"],
                "failed":  category_counts[cat]["failed"],
                "skipped": category_counts[cat]["skipped"],
            }
            for cat in CATEGORIES
        },
    }

    # ---- Save metadata ------------------------------------------------
    save_metadata(list(all_entries_map.values()), summary)
    print(f"\n  Metadata saved -> {METADATA_FILE}")

    # ---- Print final summary ------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total images found          : {total_found}")
    print(f"  Successfully processed      : {total_success}")
    print(f"  Failed (logged in metadata) : {total_failed}")
    print(f"  Already processed (skipped) : {total_skipped}")
    print()
    print("  Per-category breakdown:")
    for cat in CATEGORIES:
        cc = category_counts[cat]
        print(
            f"    {cat:<10} | found: {cc['found']:>4}  "
            f"success: {cc['success']:>4}  "
            f"failed: {cc['failed']:>4}  "
            f"skipped: {cc['skipped']:>4}"
        )
    print("=" * 60)
    print("Phase 5.2 complete.\n")


if __name__ == "__main__":
    main()
