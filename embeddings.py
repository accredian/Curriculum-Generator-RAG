from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
from transformers import pipeline
import os
from huggingface_hub import login



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

def generate_llm_output(results, max_length=500):
    """
    Generate output from LLM using retrieved results.
    :param results: List of matching curriculum records.
    :param max_length: Max length of generated text.
    :return: Generated text from LLM.
    """
    # Use the correct model identifier for Llama 2
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    
    generator = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    
    input_text = "\n\n".join([f"Subject: {res['subject']}\nCurriculum: {res['curriculum']}" for res in results])
    prompt = f"Create a detailed course outline for the following:\n{input_text}"
    
    output = generator(
        prompt,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        num_return_sequences=1
    )
    
    return output[0]['generated_text']


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
    user_query = "Strategic Management course for 12 months"

    # Run the pipeline
    output = run_rag_pipeline(json_file, user_query)
    print(output)