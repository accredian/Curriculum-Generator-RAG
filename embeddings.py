from transformers import pipeline
import os
from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
from transformers import pipeline
from huggingface_hub import login
import json
os.environ['OPENAI_API_KEY'] = 'sk-xxxxxxxxxxxxx'
os.environ['SERPER_API_KEY'] = '9f706fe3bb60606ca3a8d0cbf5b4986b31d4a84d'
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
import openai


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
    
    results = [metadata[idx] for idx in indices[0]]
    
    return results

#----------------------------------------------------------------------crew ai-----------------------------------------------------------------------------#



#-----------------------------------------------------------------------Pass result to llm-------------------------------------------------------------#



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# from g4f.client import Client

def generate_llm_output(results, max_new_tokens=200, course_name=None, duration=None, max_length=None):
    """
    Generate output from OpenAI's GPT model using CrewAI for enhanced curriculum generation.
    """
    # Initialize tool
    tool = SerperDevTool()
    
    # Prepare input text from vector DB results
    input_text = "\n\n".join([f"Subject: {res['subject']}\nCurriculum: {res['curriculum']}" for res in results])
    
    # Calculate curriculum structure
    num_terms = duration
    num_modules = duration * 4
    num_topics = duration * 4 * 4
    
    # Base prompt with structure requirements
    base_prompt = f"""
        You are tasked with designing a unique and detailed curriculum for the following course:
        
        **Course Name:** {course_name}  
        **Duration:** {duration} months  
        
        Logic for the curriculum structure:  
        - For every month, assign 1 term.  
        - Each term consists of 4 modules.  
        - Each module contains 4 topics.  
        
        Based on this structure:  
        - This course should have {num_terms} terms.  
        - It should include {num_modules} modules.  
        - A total of {num_topics} topics should be covered.  
        
        Reference curriculum for inspiration:  
        {input_text}
    """

    # Create specialized agents
    market_researcher = Agent(
        role='Market Research Specialist',
        goal='Research current industry trends and requirements',
        backstory='Expert in industry analysis with deep understanding of market demands',
        verbose=False,
        allow_delegation=True,
        tools=[tool]
    )

    curriculum_designer = Agent(
        role='Curriculum Architect',
        goal='Design curriculum following exact term-module-topic structure',
        backstory='Senior curriculum designer specializing in structured learning paths',
        verbose=False,
        allow_delegation=True,
        tools=[tool]
    )

    content_enricher = Agent(
        role='Content Enhancement Specialist',
        goal='Add practical examples while maintaining structure',
        backstory='Expert in combining theory with real-world applications',
        verbose=False,
        allow_delegation=True,
        tools=[tool]
    )

    quality_reviewer = Agent(
        role='Quality Assurance Specialist',
        goal='Ensure final curriculum is well-formatted, maintains exact structure, and meets quality standards',
        backstory='Experienced in curriculum validation and quality control with expertise in clear formatting',
        verbose=False,
        allow_delegation=True,
        tools=[tool]
    )

    # Define tasks with expected outputs
    research_task = Task(
        description=f"""
        Research current trends and requirements for {course_name}:
        1. Identify industry trends and demands
        2. Research tools and technologies
        3. Find relevant case studies

        {base_prompt}
        """,
        expected_output="A comprehensive report of current industry trends, required skills, and market demands for the course topic",
        agent=market_researcher
    )

    design_task = Task(
        description=f"""
        Design the curriculum structure using research findings:
        1. Follow the exact term-module-topic structure
        2. Define learning objectives
        3. Ensure progression logic

        {base_prompt}
        """,
        expected_output="A structured curriculum outline following the specified term-module-topic format with clear learning objectives",
        agent=curriculum_designer
    )

    enrich_task = Task(
        description=f"""
        Enhance the curriculum with practical elements:
        1. Add real-world examples
        2. Include industry tools
        3. Maintain the required structure

        {base_prompt}
        """,
        expected_output="An enhanced curriculum with practical examples, tools, and real-world applications while maintaining the required structure",
        agent=content_enricher
    )

    review_task = Task(
        description=f"""
        Review, validate, and format the final curriculum:
        1. Verify exact structure compliance ({num_terms} terms, {num_modules} modules, {num_topics} topics)
        2. Ensure clear formatting and organization
        3. Validate content quality and completeness
        4. Format the final output in a clear, well-structured manner

        The final output must be perfectly formatted and ready for direct use.

        {base_prompt}
        """,
        expected_output="A final, perfectly formatted curriculum that strictly follows the required structure and is ready for immediate use",
        agent=quality_reviewer
    )

    # Create and run the crew with updated parameters
    crew = Crew(
        agents=[market_researcher, curriculum_designer, content_enricher, quality_reviewer],
        tasks=[research_task, design_task, enrich_task, review_task],
        memory=True,
        cache=True,
        max_rpm=100,
        share_crew=True,
        process=Process.sequential
    )

    try:
        return crew.kickoff()
    
    except Exception as e:
        print(f"Error in curriculum generation: {e}")
        return None


#-----------------------------------------------------------------------Pass result to llm-------------------------------------------------------------#





# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# from g4f.client import Client

# def generate_llm_output(results, max_length=None , course_name = None, duration = None):
#     """
#     Generate output from LLM using retrieved results and GPT-4o-mini API.
#     :param results: List of matching curriculum records.
#     :param max_length: Max length of generated text.
#     :return: Generated text from LLM.
#     """
#     # Initialize the GPT-4o-mini client
#     client = Client()
    
#     # Prepare the input text based on the results
#     input_text = "\n\n".join([f"Subject: {res['subject']}\nCurriculum: {res['curriculum']}" for res in results])
#     num_terms = duration
#     num_modules = duration * 4
#     num_topics = duration * 4 * 4
#     prompt = f"""
#         You are tasked with designing a unique and detailed curriculum for the following course:
        
#         **Course Name:** {course_name}  
#         **Duration:** {duration} months  
        
#         Logic for the curriculum structure:  
#         - For every month, assign 1 term.  
#         - Each term consists of 4 modules.  
#         - Each module contains 4 topics.  
        
#         Based on this structure:  
#         - This course should have {num_terms} terms.  
#         - It should include {num_modules} modules.  
#         - A total of {num_topics} topics should be covered.  
        
#         Use the provided curriculum below as inspiration for structure, formatting, and level of detail.  
#         **Do not copy any part of the example text directly.**  
#         Instead, generate a fresh and creative curriculum tailored to the course's requirements and duration.  
#         Reference curriculum for inspiration:  
#         {input_text}  
        
#         Ensure your output is distinct, clear, and aligned with the course's focus areas, while adhering to the calculated structure.
#         """

                    
#     # Make the API call to GPT-4o\
#     response = client.chat.completions.create(
#         model="gpt-4o",  
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=max_length  # Limit the number of tokens generated
#     )
    
#     # Extract the generated text from the API response
#     generated_text = response.choices[0].message.content
    
#     return generated_text


#-----------------------------------------------------------------------Full pipeline-------------------------------------------------------------#



def run_rag_pipeline(json_file, user_query , course_name = None, duration = None):
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
    return generate_llm_output(results , course_name = course_name, duration = duration)






if __name__ == "__main__":
    # Set up Hugging Face authentication
    HF_TOKEN = "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"  
    set_hf_token(HF_TOKEN)
    
    json_file = "curriculum_data.json"

    # Example user query
    course_name = input("Enter the course name: ")
    duration = int(input("Enter the course duration: "))
    user_query = f"{course_name} course for {duration} months"

    # Run the pipeline
    output = run_rag_pipeline(json_file, user_query, course_name, duration)
    print(output)

    # # Save output to a text file
    # with open("output.txt", "w") as file:
    #     file.write(output)
