import os
import re
import json
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image
from pathlib import Path
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

def extract_month(text):
    """
    Extracts the month number from the text using multiple regex patterns.
    
    Args:
        text (str): The text to search for month number.
    
    Returns:
        str or None: The extracted month number, or None if not found.
    """
    # First pattern: "Learn what matters in X months!"
    learn_pattern = re.compile(r'Learn\s+what\s+matters\s+in\s+(\d+)\s*(?:month|months)!?', re.IGNORECASE)
    
    # Second pattern: "X Months |"
    months_pipe_pattern = re.compile(r'(\d+)\s*Months\s*\|?', re.IGNORECASE)
    
    # First, try the "Learn what matters" pattern
    learn_match = learn_pattern.search(text)
    if learn_match:
        return learn_match.group(1)
    
    # If first pattern fails, try the "X Months |" pattern
    months_pipe_match = months_pipe_pattern.search(text)
    if months_pipe_match:
        return months_pipe_match.group(1)
    
    return None

def extract_curriculum(text):
    """
    Extracts curriculum text between "Program Syllabus" and the first occurrence of delimiters.
    """
    # Case insensitive search for "Program Syllabus"
    start_pattern = re.compile(r'Program\s+Syllabus', re.IGNORECASE)
    
    # Case insensitive search for delimiters
    delimiter_pattern = re.compile(r'(world\s+class|capstone|learn\s+from)', re.IGNORECASE)
    
    # Find start position
    start_match = start_pattern.search(text)
    if not start_match:
        return ""
    
    start_pos = start_match.end()
    
    # Find end position (first occurrence of any delimiter)
    end_match = delimiter_pattern.search(text[start_pos:])
    if not end_match:
        curriculum = text[start_pos:].strip()
    else:
        curriculum = text[start_pos:start_pos + end_match.start()].strip()
    
    return curriculum


def extract_text_using_docling(pdf_path):
    """
    Attempts to extract text using Docling library
    """
    try:
        input_doc = Path(pdf_path)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
        pipeline_options.ocr_options = ocr_options
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        
        doc = converter.convert(input_doc).document
        return doc.export_to_markdown()
    except Exception as e:
        print(f"Docling extraction failed: {e}")
        return None

def extract_text_using_fallback(pdf_path):
    """
    Fallback method using PyPDF2 for text and pytesseract for images
    """
    extracted_text = ""
    
    try:
        # First try to extract text directly
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
            except Exception as e:
                print(f"Error extracting text from page: {e}")
        
        # If we got very little text, try OCR on the whole document
        if len(extracted_text.strip()) < 100:  # Arbitrary threshold
            try:
                images = convert_from_path(pdf_path)
                for i, image in enumerate(images):
                    try:
                        text = pytesseract.image_to_string(image)
                        if text.strip():
                            extracted_text += text + "\n"
                    except Exception as e:
                        print(f"Error processing image {i}: {e}")
            except Exception as e:
                print(f"Error converting PDF to images: {e}")
    
    except Exception as e:
        print(f"Fallback extraction failed: {e}")
        return None
    
    return extracted_text if extracted_text.strip() else None

def extract_text_from_pdf_with_images(pdf_path):
    """
    Extracts text from PDF using multiple methods with fallback
    """
    if not os.path.exists(pdf_path):
        return f"Error: File not found at {pdf_path}"

    # Try Docling first
    text = extract_text_using_docling(pdf_path)
    if text and len(text.strip()) > 0:
        print(f"Successfully extracted text using Docling from {pdf_path}")
        return text

    # If Docling fails, try fallback method
    print(f"Falling back to alternative extraction method for {pdf_path}")
    text = extract_text_using_fallback(pdf_path)
    if text:
        return text

    return "Error: Could not extract text using any available method"

def process_pdf_directory(directory_path):
    """
    Processes all PDFs in a directory and creates JSON output with enhanced error handling
    """
    results = []
    
    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    folder_name = os.path.basename(root)
                    
                    print(f"\nProcessing: {pdf_path}")
                    
                    # Extract text from PDF with fallback mechanisms
                    extracted_text = extract_text_from_pdf_with_images(pdf_path)
                    
                    if not extracted_text.startswith("Error"):
                        # Extract curriculum and months
                        curriculum = extract_curriculum(extracted_text)
                        months = extract_month(extracted_text)
                        
                        if curriculum:
                            result = {
                                "file_name": file,
                                "subject": folder_name,
                                "curriculum": curriculum,
                                "months": months if months else "Month not found"
                            }
                            results.append(result)
                        else:
                            print(f"No curriculum found in {file}")
                    else:
                        print(f"Extraction failed for {file}: {extracted_text}")
    
    except Exception as e:
        print(f"Error processing directory: {e}")
    
    return results


# Example usage
directory_path = "/workspaces/Curriculum_Generator"  # Replace with your directory path
results = process_pdf_directory(directory_path)

# Save to JSON file
with open('curriculum_data.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Print results
print("\nExtracted Curriculum Data:")
print(json.dumps(results, indent=2))