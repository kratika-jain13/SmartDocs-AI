SmartDocs AI Dataset
===================

Purpose
-------
This dataset provides a structured collection of business and personal document types that will be used to train and evaluate SmartDocs AI in subsequent phases.

Document Categories
------------------
- invoice
- receipt
- bank_statement
- resume
- form
- identity_document
- other

Supported Formats
-----------------
- pdf
- jpg / jpeg
- png
- docx
- txt

Folder Structure
----------------
`
data/
+- raw/
¦   +- invoices/
¦   +- receipts/
¦   +- bank_statements/
¦   +- resumes/
¦   +- forms/
¦   +- identity_documents/
¦   +- other/
+- metadata/
+- processed/
+- README.md
`

File Naming Convention
----------------------
All files should use lowercase, no spaces, and descriptive names with sequential numbering, e.g., invoice_001.pdf.

Privacy Requirements
--------------------
Do **not** use real private or confidential documents. Prefer public, synthetic, or properly anonymized data.

Licensing
----------
The dataset will be released under an open-source licence that permits research and commercial use while respecting privacy.

Future Use
-----------
Collected documents will later be processed (OCR, parsing, ML) in Phases 5-7 to build searchable, structured representations.
