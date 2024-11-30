import json
import re
def clean_curriculum_modules(raw_text):
    # Split text into modules
    modules = re.split(r'## Module \d+:', raw_text)[1:]  # Skip first empty split
    
    formatted_modules = []
    
    for module_text in modules:
        # Extract module name
        module_name = module_text.split('\n')[0].strip()
        
        # Extract topics
        topics = []
        topic_pattern = r'Topic \d+:\s*([^\n]+)'
        topic_matches = re.findall(topic_pattern, module_text)
        topics = [topic.strip() for topic in topic_matches if topic.strip()]
        
        # Format module
        formatted_module = f"Module: {module_name}\nTopics:"
        for topic in topics:
            formatted_module += f"\n- {topic}"
            
        formatted_modules.append(formatted_module)
    
    return "\n\n".join(formatted_modules)


def process_curriculum_json(json_file_path):
    # Load JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_curricula = []
    
    # Process each curriculum entry
    for entry in data:
        raw_text = entry['curriculum']
        
        # Process the text using our previous function
        formatted_curriculum = clean_curriculum_modules(raw_text)
        
        # Check if formatted curriculum is empty
        if not formatted_curriculum or formatted_curriculum.isspace():
            # Keep the original curriculum text if formatted is empty
            formatted_curriculum = raw_text if raw_text else "No curriculum content available"
        
        processed_entry = {
            "subject": entry['subject'],
            "months": entry['months'],
            "formatted_curriculum": formatted_curriculum
        }
        
        processed_curricula.append(processed_entry)
        
        # Print processing status
        print(f"Processed {entry['subject']}: {'Success' if formatted_curriculum else 'Used original'}")
    
    # Save to clean_json.json
    with open('clean_json.json', 'w', encoding='utf-8') as f:
        json.dump(processed_curricula, f, indent=2, ensure_ascii=False)
    
    print("\nTotal entries processed:", len(processed_curricula))
    print("Data saved to clean_json.json")
    return processed_curricula

# Add a function to check content validity
def validate_curriculum_entry(entry):
    curriculum = entry.get('formatted_curriculum', '')
    if not curriculum or curriculum.isspace():
        return False
    return True

# After processing, you can check which entries have content:
def check_processed_data():
    with open('clean_json.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    empty_entries = []
    for entry in data:
        if not validate_curriculum_entry(entry):
            empty_entries.append(entry['subject'])
    
    if empty_entries:
        print("\nSubjects with empty curriculum:", empty_entries)
    else:
        print("\nAll entries have curriculum content")

# Run the processing
processed_data = process_curriculum_json('curriculum_data.json')
check_processed_data()