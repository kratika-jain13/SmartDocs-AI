import os
import sys
import json

# --- 1. Import Safety ---
# Dynamically add the current directory ('scripts') to the import path
# so that running 'python scripts/main.py' from the workspace root works seamlessly.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from document_utils import (
    get_document_info,
    extract_text,
    classify_document,
    extract_fields,
    validate_document
)

def run_pipeline():
    # --- 2. Path Resolution ---
    # Find the workspace root directory (one level up from the 'scripts' folder)
    workspace_root = os.path.abspath(os.path.join(current_dir, ".."))
    raw_dir = os.path.join(workspace_root, "data", "raw")
    processed_file = os.path.join(workspace_root, "data", "processed", "processed_docs.json")

    print("==================================================")
    print("      SMARTDOCS AI - PHASE 3 DOCUMENT PIPELINE    ")
    print("==================================================\n")

    # Ensure the raw directory exists
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory '{raw_dir}' does not exist.")
        return

    # --- 3. Find Supported Files ---
    # Support: .txt, .pdf, .jpg, .jpeg, .png
    supported_extensions = (".txt", ".pdf", ".jpg", ".jpeg", ".png")
    try:
        raw_files = [
            f for f in os.listdir(raw_dir)
            if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(raw_dir, f))
        ]
    except Exception as e:
        print(f"Error scanning directory '{raw_dir}': {e}")
        return

    print(f"Found {len(raw_files)} supported documents to process: {raw_files}\n")

    # Keep track of statistics
    validation_passed_count = 0
    validation_failed_count = 0
    processing_errors_count = 0

    processed_documents = []

    # --- 4. Process Each File Resiliently ---
    for filename in raw_files:
        print(f"Processing: {filename}...")
        
        # Initialize default document metadata record
        meta = get_document_info(filename)
        doc_record = {
            "filename": filename,
            "extension": meta["extension"],
            "file_type": meta["document_type"],
            "classification": "Unknown",
            "extracted_fields": {
                "doc_id": "Unknown",
                "date": "Unknown",
                "vendor": "Unknown",
                "total_amount": 0.0
            },
            "validation": {
                "is_valid": False,
                "errors": []
            }
        }

        try:
            file_path = os.path.join(raw_dir, filename)

            # Check for non-TXT file placeholder (we only read text files for Phase 3)
            if meta["document_type"] != "Text":
                raise NotImplementedError(
                    f"OCR/Extraction not implemented for {meta['document_type']} file type in Phase 3."
                )

            # Read raw text content (standard File Handling)
            raw_text = extract_text(file_path)

            # Classify document based on content heuristics
            classification = classify_document(raw_text)
            doc_record["classification"] = classification

            # Extract fields using regex patterns
            fields = extract_fields(raw_text, classification)
            doc_record["extracted_fields"] = fields

            # Validate fields
            validation = validate_document(fields)
            doc_record["validation"] = validation

            if validation["is_valid"]:
                validation_passed_count += 1
                print(f"   - Class: {classification}")
                print(f"   - Extracted fields: {fields}")
                print(f"   - Validation: PASS")
            else:
                validation_failed_count += 1
                print(f"   - Class: {classification}")
                print(f"   - Extracted fields: {fields}")
                print(f"   - Validation: FAIL")
                print(f"     - Errors: {validation['errors']}")

        except Exception as e:
            # Handle processing failure safely without crashing the main loop
            processing_errors_count += 1
            error_message = f"Processing Error: {str(e)}"
            doc_record["validation"]["errors"].append(error_message)
            print(f"   - Status: ERROR")
            print(f"     - Reason: {error_message}")

        processed_documents.append(doc_record)
        print()

    # --- 5. Save Structured JSON ---
    print(f"Saving processing results to '{processed_file}'...")
    try:
        # Create processed folder path if it does not exist
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)

        with open(processed_file, "w", encoding="utf-8") as f:
            # Serialize the list of dictionaries to formatted JSON
            json.dump(processed_documents, f, indent=4)
        print("Success: Results saved successfully.\n")
    except Exception as e:
        print(f"Error saving results: {e}\n")

    # --- 6. Pipeline Summary Stats ---
    print("=================== SUMMARY ======================")
    print(f"Total Documents Processed: {len(processed_documents)}")
    print(f"Validation Passed: {validation_passed_count}")
    print(f"Validation Failed: {validation_failed_count}")
    print(f"Processing Errors: {processing_errors_count}")
    print("==================================================")

if __name__ == "__main__":
    # Execute the pipeline when run directly
    run_pipeline()