import os
import re
import json
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image

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
    months_pipe_pattern = re.compile(r'(\d+)\s*Months\s*\|', re.IGNORECASE)
    
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

def extract_text_from_pdf_with_images(pdf_path):
    """
    Extracts text from both text-based and image-based content in a PDF with robust error handling.
    """
    extracted_text = ""

    if not os.path.exists(pdf_path):
        return f"Error: File not found at {pdf_path}"

    try:
        # Extract text from text-based PDFs
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
            except Exception as e:
                print(f"Error processing page text: {e}")

        # Process images in the PDF for OCR
        try:
            images = convert_from_path(pdf_path)
            for image in images:
                try:
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        extracted_text += text + "\n"
                except Exception as e:
                    print(f"Error processing image: {e}")
        except Exception as e:
            print(f"Error converting PDF to images: {e}")

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return f"Error: {str(e)}"

    return extracted_text

def process_pdf_directory(directory_path):
    """
    Processes all PDFs in a directory and creates JSON output
    """
    results = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                
                print(f"Processing: {pdf_path}")
                
                # Extract text from PDF
                extracted_text = extract_text_from_pdf_with_images(pdf_path)
                
                # Extract curriculum
                curriculum = extract_curriculum(extracted_text)
                
                # Extract months
                months = extract_month(extracted_text)
                
                if curriculum:
                    result = {
                        "file_name": file,
                        "subject": folder_name,
                        "curriculum": curriculum,
                        "months": months if months else "Month not found"
                    }
                    results.append(result)
    
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