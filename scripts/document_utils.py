import os
import re

def get_document_info(filename):
    """
    Returns basic metadata about a file based on its name and extension.
    Supports .txt, .pdf, .jpg, .jpeg, and .png extensions.
    """
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".pdf":
        document_type = "PDF"
    elif extension in [".jpg", ".jpeg", ".png"]:
        document_type = "Image"
    elif extension == ".txt":
        document_type = "Text"
    else:
        document_type = "Other"

    return {
        "filename": filename,
        "extension": extension,
        "document_type": document_type
    }

def extract_text(file_path):
    """
    Reads the content of a document text file.
    In later phases, this will use actual OCR or PDF extraction libraries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def classify_document(text):
    """
    Classifies a document based on simple keyword heuristics.
    In later phases, this will use Machine Learning models (Phase 10).
    """
    text_lower = text.lower()
    if "invoice" in text_lower:
        return "Invoice"
    elif "receipt" in text_lower:
        return "Receipt"
    else:
        return "Other"

def extract_fields(text, document_type):
    """
    Extracts key fields (doc_id, date, vendor, total_amount) from text using regular expressions.
    Supports common label variations and numeric formats (including currency symbols and commas).
    """
    fields = {
        "doc_id": "Unknown",
        "date": "Unknown",
        "vendor": "Unknown",
        "total_amount": 0.0
    }
    
    # Regular expressions for line-based matching (case-insensitive)
    # Match variations like: Invoice No:, Invoice Number:, Invoice #:, Receipt No:, etc.
    doc_id_regex = re.compile(
        r'(?:invoice\s+no|invoice\s+number|invoice\s*#|receipt\s+no|receipt\s+number|receipt\s*#)\s*:\s*(.+)',
        re.IGNORECASE
    )
    
    # Match Date:
    date_regex = re.compile(r'date\s*:\s*(.+)', re.IGNORECASE)
    
    # Match Vendor:
    vendor_regex = re.compile(r'vendor\s*:\s*(.+)', re.IGNORECASE)
    
    # Match variations like: Total Amount:, Total:, Grand Total:, Amount:
    total_regex = re.compile(
        r'(?:total\s+amount|total|grand\s+total|amount)\s*:\s*(.+)',
        re.IGNORECASE
    )
    
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try matching Document ID
        doc_id_match = doc_id_regex.search(line)
        if doc_id_match:
            fields["doc_id"] = doc_id_match.group(1).strip()
            continue
            
        # Try matching Date
        date_match = date_regex.search(line)
        if date_match:
            fields["date"] = date_match.group(1).strip()
            continue
            
        # Try matching Vendor
        vendor_match = vendor_regex.search(line)
        if vendor_match:
            fields["vendor"] = vendor_match.group(1).strip()
            continue
            
        # Try matching Total Amount
        total_match = total_regex.search(line)
        if total_match:
            val_str = total_match.group(1).strip()
            # Clean currency symbols and commas out of the number
            # E.g. $2,500.00 -> 2500.00, ₹1500 -> 1500
            numeric_match = re.search(r'([\d,]+\.?\d*)', val_str)
            if numeric_match:
                clean_num_str = numeric_match.group(1).replace(",", "")
                try:
                    fields["total_amount"] = float(clean_num_str)
                except ValueError:
                    fields["total_amount"] = 0.0
            continue
            
    return fields

def validate_document(fields):
    """
    Validates that all key fields (doc_id, date, vendor, total_amount) exist and are valid.
    """
    is_valid = True
    errors = []
    
    # Check Document ID
    doc_id = fields.get("doc_id")
    if not doc_id or doc_id == "Unknown" or str(doc_id).strip() == "":
        is_valid = False
        errors.append("Missing Document ID")
        
    # Check Date
    date = fields.get("date")
    if not date or date == "Unknown" or str(date).strip() == "":
        is_valid = False
        errors.append("Missing Date")
        
    # Check Vendor
    vendor = fields.get("vendor")
    if not vendor or vendor == "Unknown" or str(vendor).strip() == "":
        is_valid = False
        errors.append("Missing Vendor")
        
    # Check Total Amount
    total_amount = fields.get("total_amount")
    if total_amount is None:
        is_valid = False
        errors.append("Missing Total Amount")
    else:
        try:
            val = float(total_amount)
            if val <= 0.0:
                is_valid = False
                errors.append("Total Amount must be greater than 0")
        except (ValueError, TypeError):
            is_valid = False
            errors.append("Invalid Total Amount format")
            
    return {
        "is_valid": is_valid,
        "errors": errors
    }