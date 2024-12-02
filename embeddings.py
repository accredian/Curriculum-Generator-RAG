from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
from transformers import pipeline
import os
from huggingface_hub import login
# os.environ['OPENAI_API_KEY'] = 'sk-lsjfd'



def set_hf_token(token):
    """
    Set up Hugging Face authentication
    :param token: Hugging Face API token
    """
    login(token=token)
    os.environ["HUGGINGFACE_TOKEN"] = token




def load_json(file_path):
    """
    Load JSON data from a file.
    :param file_path: Path to the JSON file.
    :return: Loaded JSON data as a list of dictionaries.
    """
    with open(file_path, "r") as file:
        return json.load(file)

# ---------------------------------------------------------------------------embeddings------------------------------------------------------------------------------#
def generate_embeddings(data, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for curriculum data.
    :param data: List of JSON records with curriculum field.
    :param model_name: Pre-trained model to use for embeddings.
    :return: Data with embeddings added as a new field.
    """
    model = SentenceTransformer(model_name)
    for record in data:
        curriculum_text = record["curriculum"]
        record["embedding"] = model.encode(curriculum_text).tolist()
    return data
# ---------------------------------------------------------------------------Save and Load FAISS Index------------------------------------------------------------------------------#


def create_faiss_index(data):
    """
    Create a FAISS index from curriculum embeddings.
    :param data: List of JSON records with embeddings.
    :return: FAISS index and metadata dictionary.
    """
    dimension = len(data[0]["embedding"])
    index = faiss.IndexFlatL2(dimension)

    embeddings = np.array([record["embedding"] for record in data]).astype('float32')
    index.add(embeddings)

    metadata = {i: record for i, record in enumerate(data)}
    return index, metadata



def save_faiss_index(index, metadata, index_path="index.faiss", metadata_path="metadata.json"):
    """
    Save FAISS index and metadata to some place.
    :param index: FAISS index.
    :param metadata: Metadata dictionary.
    :param index_path: Path to save the FAISS index.
    :param metadata_path: Path to save the metadata JSON.
    """
    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)



def load_faiss_index(index_path="index.faiss", metadata_path="metadata.json"):
    """
    Load FAISS index and metadata from that place.
    :param index_path: Path to FAISS index file.
    :param metadata_path: Path to metadata JSON file.
    :return: FAISS index and metadata dictionary.
    """
    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    return index, metadata


#-----------------------------------------------------------------------4. Query the Database-------------------------------------------------------------#
def query_faiss_index(index, metadata, query, model_name='all-MiniLM-L6-v2', k=5):
    """
    Query the FAISS index with user input.
    :param index: FAISS index.
    :param metadata: Metadata dictionary.
    :param query: User's query string.
    :param model_name: Pre-trained model to generate query embedding.
    :param k: Number of results to return.
    :return: List of matching records.
    """
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query).astype('float32')
    distances, indices = index.search(np.array([query_embedding]), k)
    return [metadata[idx] for idx in indices[0]]


#-----------------------------------------------------------------------Pass result to llm-------------------------------------------------------------#

# def generate_llm_output(results, max_new_tokens=500):
#     """
#     Generate output from LLM using retrieved results.
#     :param results: List of matching curriculum records.
#     :param max_new_tokens: Maximum number of new tokens to generate.
#     :return: Generated text from LLM.
#     """
#     # Use the correct model identifier for Llama 3
#     model_id = "meta-llama/Llama-3.2-1B"
    
#     generator = pipeline(
#         "text-generation",
#         model=model_id,
#         torch_dtype="auto",
#         device_map="auto"
#     )
    
#     input_text = "\n\n".join([f"Subject: {res['subject']}\nCurriculum: {res['curriculum']}" for res in results])
#     prompt = f"Create a detailed curriculum for the following according to the curriculum I am giving to you:\n{input_text}"
    
#     output = generator(
#         prompt,
#         max_new_tokens=max_new_tokens,  # Change to max_new_tokens
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.95,
#         num_return_sequences=1
#     )
    
#     return output[0]['generated_text']

# import openai

# def generate_llm_output(results, max_tokens=None):
#     """
#     Generate output from OpenAI's GPT model using retrieved results.
#     :param results: List of matching curriculum records.
#     :param max_tokens: Maximum number of tokens to generate.
#     :return: Generated text from OpenAI API.
#     """
#     # Initialize OpenAI client
#     client = openai.OpenAI()
    
#     # Prepare input text
#     input_text = "\n\n".join([f"Subject: {res['subject']}\nCurriculum: {res['curriculum']}" for res in results])
    
#     # Construct prompt
#     prompt = f"Create a detailed curriculum for the following according to the curriculum I am giving to you , carefully examine the number of pillars if there and modules and terms and try to keep the number same as given:\n{input_text}"
    
#     try:
#         # Call OpenAI API
#         response = client.chat.completions.create(
#             model="gpt-4o",  # You can change to gpt-4 if preferred
#             messages=[
#                 {"role": "system", "content": "You are a helpful curriculum design assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             # max_tokens=max_tokens,
#             n=1,
#             stop=None,
#             temperature=0.7
#         )
        
#         # Extract and return the generated text
#         return response.choices[0].message.content.strip()
    
#     except Exception as e:
#         print(f"Error in OpenAI API call: {e}")
#         return None

from g4f.client import Client

def generate_llm_output(results, max_length=None):
    """
    Generate output from LLM using retrieved results and GPT-4o-mini API.
    :param results: List of matching curriculum records.
    :param max_length: Max length of generated text.
    :return: Generated text from LLM.
    """
    # Initialize the GPT-4o-mini client
    client = Client()
    
    # Prepare the input text based on the results
    input_text = "\n\n".join([f"Subject: {res['subject']}\nCurriculum: {res['curriculum']}" for res in results])
    prompt = f"""Design a detailed curriculum based on the course duration, using the provided example curriculum as a benchmark for structure and coherence. For every month, assign 1 term. Each term consists of 4 modules, and each module contains 4 topics. Ensure the total number of terms, modules, and topics is proportional to the course duration.  

    For example:
        - A 12-month program will have 12 terms, 48 modules, and 192 topics.
        - An 8-month program will have 8 terms, 32 modules, and 128 topics.
        - A 6-month program will have 6 terms, 24 modules, and 96 topics.  

    Use the provided curriculum below as a reference for tone, style, and content distribution. Create a similarly structured curriculum, adjusting the specifics based on the course duration and ignore pillars do not use them and do not copy paste the given example curriculum:  

    {input_text}  
    """

                    
    # Make the API call to GPT-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o",  # Using GPT-4o-mini model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length  # Limit the number of tokens generated
    )
    
    # Extract the generated text from the API response
    generated_text = response.choices[0].message.content
    
    return generated_text



#-----------------------------------------------------------------------Full pipeline-------------------------------------------------------------#


def run_rag_pipeline(json_file, user_query):
    """
    End-to-end RAG pipeline to process curriculum data and generate results.
    :param json_file: Path to JSON file containing curriculum data.
    :param user_query: Query provided by the user.
    :return: Generated output from LLM.
    """
    # Step 1: Load data
    data = load_json(json_file)

    # Step 2: Generate embeddings
    data = generate_embeddings(data)

    # Step 3: Create and save FAISS index
    index, metadata = create_faiss_index(data)
    save_faiss_index(index, metadata)

    # Step 4: Query the database
    results = query_faiss_index(index, metadata, user_query)

    # Step 5: Generate LLM output
    return generate_llm_output(results)

if __name__ == "__main__":
    # Set up Hugging Face authentication
    HF_TOKEN = "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"  # Replace with your actual token
    set_hf_token(HF_TOKEN)
    
    # File paths
    json_file = "curriculum_data.json"

    # Example user query
    user_query = "Project Management course for 6 months"

    # Run the pipeline
    output = run_rag_pipeline(json_file, user_query)
    print(output)