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
os.environ['OPENAI_API_KEY'] = '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'

os.environ['SERPER_API_KEY'] = '9f706fe3bb60606ca3a8d0cbf5b4986b31d4a84d'
# Must precede any llm module imports

from langtrace_python_sdk import langtrace

langtrace.init(api_key = '56acaaf0e99005bab5ad6088ab368d2cfa96cf9e507aee506e58abf9c352f1fa')
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileWriterTool, SerperDevTool
import streamlit as st
from markdown import markdown
from weasyprint import HTML
import tempfile
import time
from typing import Dict, Any, Tuple
from functools import wraps
import pandas as pd


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
def generate_llm_output(results, max_new_tokens=200, course_name=None, duration=None, duration_type="months", max_length=None):
    """
    Generate output from OpenAI's GPT model using CrewAI for enhanced curriculum generation.
    Modified to handle both duration types.
    """
    tool = SerperDevTool()
    file_writer_tool = FileWriterTool()
    
    input_text = "\n\n".join([f"Subject: {res['subject']}\nCurriculum: {res['curriculum']}" for res in results])
    
    # Calculate curriculum structure based on duration type
    if duration_type == "months":
        num_terms = duration
        num_modules = duration * 4
        num_topics = duration * 4 * 4
        
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
    else:  # hours
        num_modules = duration // 3  # Every 3 hours = 1 module
        num_terms = num_modules // 4  # Every 4 modules = 1 term
        num_topics = num_modules * 4  # Each module has 4 topics
        
        base_prompt = f"""
        You are tasked with designing a unique and detailed curriculum for the following course:
        
        **Course Name:** {course_name}  
        **Duration:** {duration} hours  
        
        Logic for the curriculum structure:  
        - Every 3 hours of content makes up 1 module.
        - Every 4 modules (12 hours) constitutes 1 term.
        - Each module contains 4 focused topics.
        
        Based on this structure:  
        - This course will have {num_terms} terms.
        - It will include {num_modules} modules (1 module per 3 hours).
        - A total of {num_topics} topics will be covered.
        
        Additional considerations:
        - Each 3-hour module should be self-contained and achievable within the time constraint.
        - Topics should be carefully scoped to fit within their allocated module time.
        - Ensure practical exercises and hands-on activities are appropriately timed.
        
        Reference curriculum for inspiration:  
        {input_text}
        """


    # Merged Agent 1: Research + Curriculum Design
    curriculum_developer = Agent(
        role='Curriculum Developer',
        goal='Research trends and design the structured curriculum',
        backstory='A senior expert in curriculum development with a strong understanding of market demands and structured learning design.',
        verbose=False,
        allow_delegation=True,
        tools=[tool]
    )

    # Merged Agent 2: Content Enhancement + Quality Review
    content_specialist = Agent(
        role='Quality Specialist',
        goal='Enhance,  ensure quality & formatting , make sure that you Structure the curriculum with exact with  ({num_terms} terms, {num_modules} modules, {num_topics} topics)',
        backstory='An experienced content designer skilled in making learning engaging while maintaining structure and clarity.',
        verbose=False,
        allow_delegation=True,
        tools=[tool, file_writer_tool]
    )

    # Task 1: Research + Curriculum Design
    curriculum_task = Task(
        description=f"""
        Research and design a curriculum for {course_name}:
        1. Identify industry trends and skills needed
        2. Research tools and technologies
        3. Structure curriculum with exact ({num_terms} terms, {num_modules} modules, {num_topics} topics)
        4. Define learning objectives and progression logic

        {base_prompt}
        """,
        expected_output="A structured curriculum following the specified term-module-topic format with clear learning objectives.",
        agent=curriculum_developer
    )

    # Task 2: Content Enhancement + Quality Review + Save to File
    final_review_task = Task(
        description=f"""
        Enhance and finalize the curriculum:
        1. Add industry-relevant case studies at the end 
        2. Ensure curriculum structure remains intact
        3. Verify correct formatting and content clarity
        4. Format in Markdown and save to 'outputs/curriculum_{course_name.lower().replace(" ", "_")}.md'
        5. make sure that you Structure the curriculum with exact with  ({num_terms} terms, {num_modules} modules, {num_topics} topics)

        The final output must be polished and ready for direct use.

        {base_prompt}
        """,
        expected_output="A final, formatted curriculum with practical examples saved as a Markdown file.",
        agent=content_specialist,
        output_file=f'outputs/curriculum_{course_name.lower().replace(" ", "_")}.md',
        create_directory=True
    )

    # Create and run the optimized crew
    crew = Crew(
        agents=[curriculum_developer, content_specialist],
        tasks=[curriculum_task, final_review_task],
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


# @timing_decorator
# def run_rag_pipeline(json_file, user_query, course_name=None, duration=None, duration_type="months"):
#     """
#     End-to-end RAG pipeline to process curriculum data and generate results.
#     Modified to handle duration type.
#     """
#     data = load_json(json_file)
#     data = generate_embeddings(data)
#     index, metadata = create_faiss_index(data)
#     save_faiss_index(index, metadata)
#     results = query_faiss_index(index, metadata, user_query)
#     return generate_llm_output(results, course_name=course_name, duration=duration, duration_type=duration_type)


@timing_decorator
def run_rag_pipeline(json_file, user_query, course_name=None, duration=None , duration_type = "months") -> Tuple[Any, Dict[str, float]]:
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
    output = generate_llm_output(results, course_name=course_name, duration=duration, duration_type=duration_type)
    
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






# Add a function to convert Markdown to PDF
@timing_decorator
def md_to_pdf(markdown_file, pdf_file):
    """
    Convert a Markdown file to a PDF file.
    :param markdown_file: Path to the input Markdown file.
    :param pdf_file: Path to the output PDF file.
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    html_content = markdown(markdown_content)

    html_complete = f"""
    <html>
        <head>
            <meta charset="utf-8">
        </head>
        <body>
            {html_content}
        </body>
    </html>
    """

    HTML(string=html_complete).write_pdf(pdf_file)



# st.title("Curriculum Generator")
# st.sidebar.header("Please provide course details")

# course_name = st.sidebar.text_input("Course Name")

# # Add radio button for duration type selection
# duration_type = st.sidebar.radio("Select Duration Type", ["Months", "Hours"])

# # Conditional input based on duration type
# if duration_type == "Months":
#     duration = st.sidebar.number_input("Duration (Months)", min_value=1, step=1)
# else:
#     duration = st.sidebar.number_input("Duration (Hours)", min_value=3, step=3)  # Minimum 3 hours (1 module)

# if st.button("Generate Curriculum"):
#     if course_name and duration:
#         with st.spinner("Processing..."):
#             json_file = "curriculum_data.json"
#             # Modify query based on duration type
#             user_query = f"{course_name} course for {duration} {duration_type.lower()}"
#             output = run_rag_pipeline(
#                 json_file, 
#                 user_query, 
#                 course_name=course_name, 
#                 duration=duration,
#                 duration_type=duration_type.lower()
#             )

#             if output:
#                 markdown_file = f'outputs/curriculum_{course_name.lower().replace(" ", "_")}.md'
#                 pdf_file = f'outputs/curriculum_{course_name.lower().replace(" ", "_")}.pdf'

#                 # Save markdown content to file
#                 with open(markdown_file, 'w', encoding='utf-8') as f:
#                     f.write(str(output))

#                 # Convert to PDF
#                 md_to_pdf(markdown_file, pdf_file)

#                 st.success("Curriculum Generated!")
                
#                 # Display rendered markdown
#                 st.markdown("### Generated Curriculum (Markdown)")
#                 st.markdown(str(output))

#                 # Provide PDF download
#                 with open(pdf_file, "rb") as f:
#                     pdf_data = f.read()
#                 st.download_button(
#                     label="Download Curriculum as PDF",
#                     data=pdf_data,
#                     file_name=f"{course_name}_curriculum.pdf",
#                     mime="application/pdf"
#                 )
#             else:
#                 st.error("Failed to generate curriculum.")
#     else:
#         st.warning("Please provide all required inputs.")


st.title("Curriculum Generator")
st.sidebar.header("Please provide course details")

course_name = st.sidebar.text_input("Course Name")

# Add radio button for duration type selection
duration_type = st.sidebar.radio("Select Duration Type", ["Months", "Hours"])

# Conditional input based on duration type
if duration_type == "Months":
    duration = st.sidebar.number_input("Duration (Months)", min_value=1, step=1)
else:
    duration = st.sidebar.number_input("Duration (Hours)", min_value=3, step=3)  # Minimum 3 hours (1 module)

if st.button("Generate Curriculum"):
    if course_name and duration:
        with st.spinner("Processing..."):
            json_file = "curriculum_data.json"
            # Modify query based on duration type
            user_query = f"{course_name} course for {duration} {duration_type.lower()}"
            
            # Capture both output and timing results
            output, timing_results = run_rag_pipeline(
                json_file, 
                user_query, 
                course_name=course_name, 
                duration=duration,
                duration_type=duration_type.lower()
            )

            if output:
                markdown_file = f'outputs/curriculum_{course_name.lower().replace(" ", "_")}.md'
                pdf_file = f'outputs/curriculum_{course_name.lower().replace(" ", "_")}.pdf'

                # Save markdown content to file
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(str(output))

                # Convert to PDF
                md_to_pdf(markdown_file, pdf_file)

                st.success("Curriculum Generated!")
                
                # Display timing results in an expander
                with st.expander("View Processing Times"):
                    st.write("### Processing Times")
                    # Create a DataFrame for better visualization
                    timing_df = pd.DataFrame({
                        'Step': timing_results.keys(),
                        'Time (seconds)': [f"{time:.2f}" for time in timing_results.values()]
                    })
                    st.dataframe(timing_df)
                    
                    # Optional: Add a bar chart
                    st.bar_chart(timing_df.set_index('Step')['Time (seconds)'].astype(float))
                
                # Display rendered markdown
                st.markdown("### Generated Curriculum (Markdown)")
                st.markdown(str(output))

                # Provide PDF download
                with open(pdf_file, "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    label="Download Curriculum as PDF",
                    data=pdf_data,
                    file_name=f"{course_name}_curriculum.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Failed to generate curriculum.")
    else:
        st.warning("Please provide all required inputs.")
