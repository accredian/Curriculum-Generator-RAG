from transformers import pipeline
import os
from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
from transformers import pipeline
from huggingface_hub import login
import json
# os.environ['OPENAI_API_KEY'] = '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'
os.environ['OPENAI_API_KEY'] = 'sk-cGNL7dFWnZchBHtALgJhT3BlbkFJ8eDktSCI6gwbeSew8DLichro'
os.environ['SERPER_API_KEY'] = '9f706fe3bb60606ca3a8d0cbf5b4986b31d4a84d'
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileWriterTool, SerperDevTool

import time
from typing import Dict, Any, Tuple
from functools import wraps

def timing_decorator(func):
    """Decorator to measure execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        wrapper.timing = execution_time  # Store timing in the function object
        print(f"{func.__name__} took {execution_time:.2f} seconds to execute")
        return result
    return wrapper

@timing_decorator
def set_hf_token(token):
    """
    Set up Hugging Face authentication
    :param token: Hugging Face API token
    """
    login(token=token)
    os.environ["HUGGINGFACE_TOKEN"] = token



@timing_decorator
def load_json(file_path):
    """
    Load JSON data from a file.
    :param file_path: Path to the JSON file.
    :return: Loaded JSON data as a list of dictionaries.
    """
    with open(file_path, "r") as file:
        return json.load(file)

# ---------------------------------------------------------------------------embeddings------------------------------------------------------------------------------#
@timing_decorator
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

@timing_decorator
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


@timing_decorator
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


@timing_decorator
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
@timing_decorator
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
@timing_decorator
def generate_llm_output(results, max_new_tokens=200, course_name=None, duration=None, max_length=None):
    """
    Generate output from OpenAI's GPT model using CrewAI for enhanced curriculum generation.
    Includes file saving functionality.
    """
    # Initialize tools
    tool = SerperDevTool()
    file_writer_tool = FileWriterTool()
    
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

    # Create specialized agents (previous agent definitions remain the same)
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
        tools=[tool, file_writer_tool]  # Add file_writer_tool to the quality reviewer
    )

    # Modified tasks with file saving
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

    # Modified review task to include file saving
    review_task = Task(
        description=f"""
        Review, validate, format the final curriculum, and save to file:
        1. Verify exact structure compliance ({num_terms} terms, {num_modules} modules, {num_topics} topics)
        2. Ensure clear formatting and organization
        3. Validate content quality and completeness
        4. Format the final output in Markdown
        5. Save the curriculum to 'outputs/curriculum_{course_name.lower().replace(" ", "_")}.md'

        The final output must be perfectly formatted and ready for direct use.

        {base_prompt}
        """,
        expected_output="A final, perfectly formatted curriculum saved as a Markdown file",
        agent=quality_reviewer,
        output_file=f'outputs/curriculum_{course_name.lower().replace(" ", "_")}.md',
        create_directory=True
    )

    # Create and run the crew
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
        result = crew.kickoff()
        
        
        return result
    
    except Exception as e:
        print(f"Error in curriculum generation: {e}")
        return None






#-----------------------------------------------------------------------Full pipeline-------------------------------------------------------------#


# def run_rag_pipeline(json_file, user_query , course_name = None, duration = None):
#     """
#     End-to-end RAG pipeline to process curriculum data and generate results.
#     :param json_file: Path to JSON file containing curriculum data.
#     :param user_query: Query provided by the user.
#     :return: Generated output from LLM.
#     """
#     # Step 1: Load data
#     data = load_json(json_file)

#     # Step 2: Generate embeddings
#     data = generate_embeddings(data)

#     # Step 3: Create and save FAISS index
#     index, metadata = create_faiss_index(data)
#     save_faiss_index(index, metadata)

#     # Step 4: Query the database
#     results = query_faiss_index(index, metadata, user_query)  

#     # Step 5: Generate LLM output
#     return generate_llm_output(results , course_name = course_name, duration = duration)
@timing_decorator
def run_rag_pipeline(json_file, user_query, course_name=None, duration=None) -> Tuple[Any, Dict[str, float]]:
    """
    End-to-end RAG pipeline with timing measurements.
    Returns a tuple of (output, timing_results)
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
    output = generate_llm_output(results, course_name=course_name, duration=duration)
    
    # Collect all timing results
    timing_results = {
        'load_json': load_json.timing,
        'generate_embeddings': generate_embeddings.timing,
        'create_faiss_index': create_faiss_index.timing,
        'save_faiss_index': save_faiss_index.timing,
        'query_faiss_index': query_faiss_index.timing,
        'generate_llm_output': generate_llm_output.timing,
        'total_time': sum([
            load_json.timing,
            generate_embeddings.timing,
            create_faiss_index.timing,
            save_faiss_index.timing,
            query_faiss_index.timing,
            generate_llm_output.timing
        ])
    }
    
    return output, timing_results





if __name__ == "__main__":
    # Set up Hugging Face authentication
    HF_TOKEN = "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"
    set_hf_token(HF_TOKEN)
    
    json_file = "curriculum_data.json"
    
    # Example user query
    course_name = input("Enter the course name: ")
    duration = int(input("Enter the course duration: "))
    user_query = f"{course_name} course for {duration} months"
    
    # Run the pipeline and get timing results
    output, timing_results = run_rag_pipeline(json_file, user_query, course_name, duration)
    
    # Print detailed timing results
    print("\nTiming Results:")
    print("-" * 50)
    for step, time_taken in timing_results.items():
        print(f"{step}: {time_taken:.2f} seconds")
