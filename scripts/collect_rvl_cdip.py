import os
import json
from collections import defaultdict

from datasets import load_dataset


# ============================================================
# SMARTDOCS AI - PHASE 4.4
# RVL-CDIP DATASET COLLECTION TEST
# ============================================================

DATASET_NAME = "hf-tuner/rvl-cdip-document-classification"

# Test collection limits
TARGETS = {
    "invoices": 500,
    "forms": 500,
    "resumes": 500,
    "other": 500
}
# RVL-CDIP labels that we want to use as "Other"
OTHER_LABELS = {
    "letter",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "presentation",
    "questionnaire",
    "memo"
}

# Project directories
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")

METADATA_FILE = os.path.join(
    METADATA_DIR,
    "collected_documents.json"
)


def create_directories():
    """Create the required dataset directories."""

    for category in TARGETS:
        category_dir = os.path.join(RAW_DIR, category)
        os.makedirs(category_dir, exist_ok=True)

    os.makedirs(METADATA_DIR, exist_ok=True)


def get_smartdocs_category(label):
    """
    Convert an RVL-CDIP label into a SmartDocs AI category.
    """

    label = str(label).strip().lower()

    if label == "invoice":
        return "invoices"

    if label == "form":
        return "forms"

    if label == "resume":
        return "resumes"

    if label in OTHER_LABELS:
        return "other"

    return None


def save_image(image, category, index):
    """
    Save one dataset image to the appropriate SmartDocs folder.
    """

    filename = f"{category}_{index:04d}.png"

    output_path = os.path.join(
        RAW_DIR,
        category,
        filename
    )

    # Hugging Face image objects can be converted to PIL images.
    image.save(output_path)

    return output_path


def main():

    print("=" * 60)
    print("     SMARTDOCS AI - PHASE 4.4 DATASET COLLECTION")
    print("=" * 60)

    print("\nTest mode enabled.")
    print("Target: 5 documents per category")
    print("Expected maximum: 20 documents\n")

    create_directories()

    print("Loading dataset from Hugging Face...")
    print(f"Dataset: {DATASET_NAME}\n")

    try:
        dataset = load_dataset(
            DATASET_NAME,
            split="train",
            streaming=True
        )
    except Exception as e:
        print("\nERROR: Could not load dataset.")
        print(f"Reason: {e}")
        return

    collected_counts = defaultdict(int)
    metadata_records = []

    print("Dataset loaded successfully.")
    print("Starting collection...\n")

    for sample in dataset:

        image = sample.get("image")
        label = sample.get("label")

        if image is None or label is None:
            continue

        # Convert numerical labels to label names if necessary.
        if isinstance(label, int):

            label_names = [
                "letter",
                "form",
                "email",
                "handwritten",
                "advertisement",
                "scientific report",
                "scientific publication",
                "specification",
                "file folder",
                "news article",
                "budget",
                "invoice",
                "presentation",
                "questionnaire",
                "resume",
                "memo"
            ]

            if 0 <= label < len(label_names):
                label = label_names[label]
            else:
                continue

        smartdocs_category = get_smartdocs_category(label)

        if smartdocs_category is None:
            continue

        # Stop collecting this category once target is reached.
        if collected_counts[smartdocs_category] >= TARGETS[smartdocs_category]:
            continue

        collected_counts[smartdocs_category] += 1

        index = collected_counts[smartdocs_category]

        try:

            output_path = save_image(
                image,
                smartdocs_category,
                index
            )

            record = {
                "filename": os.path.basename(output_path),
                "category": smartdocs_category,
                "source_dataset": DATASET_NAME,
                "original_label": str(label),
                "split": "train"
            }

            metadata_records.append(record)

            print(
                f"[{smartdocs_category.upper()}] "
                f"{index}/{TARGETS[smartdocs_category]} "
                f"-> {record['filename']}"
            )

        except Exception as e:

            print(
                f"ERROR saving {smartdocs_category} "
                f"document #{index}: {e}"
            )

            collected_counts[smartdocs_category] -= 1

        # Check whether all categories reached their target.
        if all(
            collected_counts[category] >= TARGETS[category]
            for category in TARGETS
        ):
            break

    # ========================================================
    # SAVE COLLECTION METADATA
    # ========================================================

    try:

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata_records,
                file,
                indent=4
            )

    except Exception as e:

        print("\nERROR saving metadata:")
        print(e)
        return

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("             COLLECTION TEST COMPLETE")
    print("=" * 60)

    total = 0

    for category in TARGETS:

        count = collected_counts[category]

        total += count

        print(
            f"{category.capitalize():<12}: "
            f"{count}/{TARGETS[category]}"
        )

    print("-" * 60)
    print(f"Total collected: {total}")
    print(f"Metadata saved: {METADATA_FILE}")

    print("=" * 60)

    if total == sum(TARGETS.values()):

        print("\nSUCCESS!")
        print(
            "All test documents were collected successfully."
        )

    else:

        print("\nWARNING!")
        print(
            "The target was not completely reached."
        )

        print(
            "Do NOT increase the dataset size yet."
        )


if __name__ == "__main__":
    main()