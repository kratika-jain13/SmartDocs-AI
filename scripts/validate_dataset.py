import os
import json
from collections import Counter
from PIL import Image


# ============================================================
# SMARTDOCS AI - PHASE 4.6
# DATASET VALIDATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "collected_documents.json"
)

CATEGORIES = [
    "invoices",
    "forms",
    "resumes",
    "other"
]


def load_metadata():
    """Load collected document metadata."""

    if not os.path.exists(METADATA_FILE):
        print("ERROR: Metadata file not found.")
        return []

    try:
        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception as e:
        print(f"ERROR reading metadata: {e}")
        return []


def check_category_counts():
    """Check number of files in every category."""

    counts = {}

    for category in CATEGORIES:

        category_dir = os.path.join(
            RAW_DIR,
            category
        )

        if not os.path.exists(category_dir):
            counts[category] = 0
            continue

        files = [
            file
            for file in os.listdir(category_dir)
            if os.path.isfile(
                os.path.join(category_dir, file)
            )
        ]

        counts[category] = len(files)

    return counts


def check_image_files():
    """
    Check whether every image can actually be opened.
    Detects corrupted or invalid image files.
    """

    corrupted_files = []
    valid_images = 0
    image_dimensions = []

    for category in CATEGORIES:

        category_dir = os.path.join(
            RAW_DIR,
            category
        )

        if not os.path.exists(category_dir):
            continue

        for filename in os.listdir(category_dir):

            file_path = os.path.join(
                category_dir,
                filename
            )

            if not os.path.isfile(file_path):
                continue

            try:

                with Image.open(file_path) as image:

                    image.verify()

                # Reopen to safely read dimensions
                with Image.open(file_path) as image:

                    width, height = image.size

                    image_dimensions.append(
                        (width, height)
                    )

                valid_images += 1

            except Exception as e:

                corrupted_files.append(
                    {
                        "file": file_path,
                        "error": str(e)
                    }
                )

    return (
        valid_images,
        corrupted_files,
        image_dimensions
    )


def check_duplicate_filenames():
    """Check for duplicate filenames across metadata."""

    metadata = load_metadata()

    filenames = [
        record.get("filename")
        for record in metadata
        if record.get("filename")
    ]

    counts = Counter(filenames)

    duplicates = [
        filename
        for filename, count in counts.items()
        if count > 1
    ]

    return duplicates


def check_metadata_files():
    """
    Check that every metadata record points to
    an existing physical file.
    """

    metadata = load_metadata()

    missing_files = []

    for record in metadata:

        filename = record.get("filename")
        category = record.get("category")

        if not filename or not category:
            missing_files.append(
                {
                    "filename": filename,
                    "category": category
                }
            )
            continue

        file_path = os.path.join(
            RAW_DIR,
            category,
            filename
        )

        if not os.path.exists(file_path):

            missing_files.append(
                {
                    "filename": filename,
                    "category": category
                }
            )

    return missing_files


def calculate_dimensions(image_dimensions):
    """Calculate basic image dimension statistics."""

    if not image_dimensions:
        return {
            "min_width": 0,
            "max_width": 0,
            "min_height": 0,
            "max_height": 0
        }

    widths = [
        width
        for width, height in image_dimensions
    ]

    heights = [
        height
        for width, height in image_dimensions
    ]

    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights)
    }


def main():

    print("=" * 60)
    print("        SMARTDOCS AI - DATASET VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Category counts
    # --------------------------------------------------------

    print("\n1. CATEGORY COUNTS")
    print("-" * 60)

    category_counts = check_category_counts()

    total_files = 0

    for category in CATEGORIES:

        count = category_counts[category]

        total_files += count

        print(
            f"{category.capitalize():<15}: {count}"
        )

    print("-" * 60)
    print(f"Total files      : {total_files}")

    # --------------------------------------------------------
    # 2. Metadata count
    # --------------------------------------------------------

    print("\n2. METADATA CHECK")
    print("-" * 60)

    metadata = load_metadata()

    print(
        f"Metadata records : {len(metadata)}"
    )

    metadata_match = (
        len(metadata) == total_files
    )

    print(
        f"Count matches    : "
        f"{'PASS' if metadata_match else 'FAIL'}"
    )

    # --------------------------------------------------------
    # 3. Image validation
    # --------------------------------------------------------

    print("\n3. IMAGE VALIDATION")
    print("-" * 60)

    (
        valid_images,
        corrupted_files,
        image_dimensions
    ) = check_image_files()

    print(
        f"Valid images     : {valid_images}"
    )

    print(
        f"Corrupted images : {len(corrupted_files)}"
    )

    if corrupted_files:

        print("\nCorrupted files:")

        for item in corrupted_files[:20]:

            print(
                f"  - {item['file']}"
            )

    # --------------------------------------------------------
    # 4. Duplicate filenames
    # --------------------------------------------------------

    print("\n4. DUPLICATE CHECK")
    print("-" * 60)

    duplicates = check_duplicate_filenames()

    print(
        f"Duplicate names  : {len(duplicates)}"
    )

    if duplicates:

        for filename in duplicates[:20]:

            print(
                f"  - {filename}"
            )

    # --------------------------------------------------------
    # 5. Missing files
    # --------------------------------------------------------

    print("\n5. METADATA → FILE CHECK")
    print("-" * 60)

    missing_files = check_metadata_files()

    print(
        f"Missing files    : {len(missing_files)}"
    )

    if missing_files:

        for item in missing_files[:20]:

            print(
                f"  - {item}"
            )

    # --------------------------------------------------------
    # 6. Image dimensions
    # --------------------------------------------------------

    print("\n6. IMAGE DIMENSIONS")
    print("-" * 60)

    dimension_stats = calculate_dimensions(
        image_dimensions
    )

    print(
        f"Width range      : "
        f"{dimension_stats['min_width']} - "
        f"{dimension_stats['max_width']} px"
    )

    print(
        f"Height range     : "
        f"{dimension_stats['min_height']} - "
        f"{dimension_stats['max_height']} px"
    )

    # --------------------------------------------------------
    # 7. Final validation
    # --------------------------------------------------------

    dataset_valid = (
        total_files == 2000
        and len(metadata) == 2000
        and valid_images == total_files
        and len(corrupted_files) == 0
        and len(duplicates) == 0
        and len(missing_files) == 0
    )

    print("\n" + "=" * 60)
    print("              FINAL DATASET STATUS")
    print("=" * 60)

    if dataset_valid:

        print("\nDATASET VALIDATION: PASS")

        print(
            "\nThe dataset passed all automated "
            "validation checks."
        )

    else:

        print("\nDATASET VALIDATION: FAIL")

        print(
            "\nOne or more validation checks "
            "need attention."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()