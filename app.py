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
# # os.environ['OPENAI_API_KEY'] = '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'

# os.environ['SERPER_API_KEY'] = '9f706fe3bb60606ca3a8d0cbf5b4986b31d4a84d'
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
        goal='Design curriculum following exact term-module-topic structure the curriculum that you design should ({num_terms} terms, {num_modules} modules, {num_topics} topics)',
        backstory='Senior curriculum designer specializing in structured learning paths',
        verbose=False,
        allow_delegation=True,
        tools=[tool]
    )

    # content_enricher = Agent(
    #     role='Content Enhancement Specialist',
    #     goal='Add practical examples while maintaining structure',
    #     backstory='Expert in combining theory with real-world applications',
    #     verbose=False,
    #     allow_delegation=True,
    #     tools=[tool]
    # )

    quality_reviewer = Agent(
        role='Quality Assurance Specialist',
        goal='Ensure final curriculum is well-formatted, maintains exact structure, and meets quality standards ',
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
        4. the curriculum that you design should ({num_terms} terms, {num_modules} modules, {num_topics} topics)

        {base_prompt}

        
        """,
        expected_output="A structured curriculum outline following the specified term-module-topic format with clear learning objectives",
        agent=curriculum_designer
    )

    # enrich_task = Task(
    #     description=f"""
    #     Enhance the curriculum with practical elements:
    #     1. Add real-world examples
    #     2. Include industry tools
    #     3. Maintain the required structure

    #     {base_prompt}
    #     """,
    #     expected_output="An enhanced curriculum with practical examples, tools, and real-world applications while maintaining the required structure",
    #     agent=content_enricher
    # )

    # Modified review task to include file saving
    review_task = Task(
        description=f"""
        Review, validate, format the final curriculum, and save to file:
        1. Verify exact structure compliance ({num_terms} terms, {num_modules} modules, {num_topics} topics)
        2. Ensure clear formatting and organization
        3. Validate content quality and completeness, make sure each term has 4 modules
        4. Format the final output in well structured format
        5. Save the curriculum to 'outputs/curriculum_{course_name.lower().replace(" ", "_")}.txt'

        The final output must be perfectly formatted and ready for direct use.

        {base_prompt}
        """,
        expected_output="A final, perfectly formatted curriculum saved as a txt file",
        agent=quality_reviewer,
        output_file=f'outputs/curriculum_{course_name.lower().replace(" ", "_")}.txt',
        create_directory=True
    )

    # Create and run the crew
    crew = Crew(
        agents=[market_researcher, curriculum_designer, quality_reviewer],
        tasks=[research_task, design_task, review_task],
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
def txt_to_pdf(text_file, pdf_file):
    """
    Convert a plain text/markdown file to a properly formatted PDF file.
    
    :param text_file: Path to the input text file
    :param pdf_file: Path to the output PDF file
    """
    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    # Convert Markdown to HTML with extra features enabled
    html_content = markdown(
        text_content,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.codehilite',
            'markdown.extensions.nl2br'
        ]
    )
    
    # Enhanced HTML template with better styling
    html_complete = f"""
    <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                    line-height: 1.6;
                    margin: 40px;
                    color: #333;
                }}
                h1 {{ font-size: 28px; margin-bottom: 20px; }}
                h2 {{ font-size: 24px; margin-bottom: 15px; }}
                h3 {{ font-size: 20px; margin-bottom: 10px; }}
                p {{ margin-bottom: 15px; }}
                code {{
                    background-color: #f5f5f5;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-family: monospace;
                }}
                pre {{
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                blockquote {{
                    border-left: 4px solid #ddd;
                    padding-left: 15px;
                    margin-left: 0;
                    color: #666;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 15px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
    </html>
    """
    
    # Convert to PDF with better options
    HTML(string=html_complete).write_pdf(
        pdf_file,
        presentational_hints=True
    )




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

#             # Capture both output and timing results
#             output, timing_results = run_rag_pipeline(
#                 json_file, 
#                 user_query, 
#                 course_name=course_name, 
#                 duration=duration,
#                 duration_type=duration_type.lower()
#             )

#             if output:
#                 text_file = f'outputs/curriculum{course_name.lower().replace(" ", "")}.txt'
#                 pdf_file = f'outputs/curriculum{course_name.lower().replace(" ", "")}.pdf'

#                 cleaned_output = str(output).strip().replace("```", "").strip()

#                 # Save text content to file
#                 with open(text_file, 'w', encoding='utf-8') as f:
#                     f.write(str(cleaned_output))

#                 # Convert to PDF
#                 txt_to_pdf(text_file, pdf_file)

#                 st.success("Curriculum Generated!")

#                 # Display timing results in an expander
#                 with st.expander("View Processing Times"):
#                     st.write("### Processing Times")
#                     # Create a DataFrame for better visualization
#                     timing_df = pd.DataFrame({
#                         'Step': timing_results.keys(),
#                         'Time (seconds)': [f"{time:.2f}" for time in timing_results.values()]
#                     })
#                     st.dataframe(timing_df)

#                     # Optional: Add a bar chart
#                     st.bar_chart(timing_df.set_index('Step')['Time (seconds)'].astype(float))

#                 # Display the plain text content from the text file
#                 with open(text_file, "r", encoding="utf-8") as f:
#                     text_content = f.read()

#                 st.markdown(text_content)  # Display plain text properly in Streamlit

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

# import streamlit as st
# import os
# import pandas as pd

# Add title and about section
st.title("Curriculum Generator")

# About section in an expander
# with st.expander("About", expanded=True):
#     st.markdown("""
#     ### Welcome to the Curriculum Generator!
    
#     This tool helps you generate comprehensive course curricula using AI technology. Simply provide your course details and API keys to get started.
    
#     #### How to Use:
#     1. Enter your API keys in the sidebar (required only once per session)
#     2. Input your course name
#     3. Select duration type (Months or Hours)
#     4. Specify the duration
#     5. Click 'Generate Curriculum' to create your customized course outline
    
#     #### Features:
#     - Generates detailed course structure
#     - Provides downloadable PDF output
#     - Shows processing times and performance metrics
#     - Supports both time-based and hour-based course planning
    
#     #### Note:
#     Make sure you have valid API keys for:
#     - OpenAI API
#     - Serper API
#     """)

with st.sidebar.expander("About", expanded=True):
    st.markdown("""
    ### Welcome to the Curriculum Generator!
This tool helps you generate comprehensive course curricula using AI technology. Simply provide your course details and API keys to get started.
    """)

# API Keys section in sidebar
st.sidebar.header("API Configuration")

# Store API keys in session state if not already present
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None
if 'serper_api_key' not in st.session_state:
    st.session_state.serper_api_key = None

with st.sidebar.expander("Enter API Keys", expanded=True):
    openai_key = st.text_input("OpenAI API Key", type="password")
    serper_key = st.text_input("Serper API Key", type="password")
    
    if st.button("Save API Keys"):
        if openai_key and serper_key:
            # Store in session state
            st.session_state.openai_api_key = openai_key
            st.session_state.serper_api_key = serper_key
            
            # Set in OS environment variables
            os.environ['OPENAI_API_KEY'] = openai_key
            os.environ['SERPER_API_KEY'] = serper_key
            
            st.success("API keys saved successfully!")
        else:
            st.error("Please provide both API keys.")

# Course Details Section
st.sidebar.header("Please provide course details")

course_name = st.sidebar.text_input("Course Name")

# Add radio button for duration type selection
duration_type = st.sidebar.radio("Select Duration Type", ["Months", "Hours"])

# Conditional input based on duration type
if duration_type == "Months":
    duration = st.sidebar.number_input("Duration (Months)", min_value=1, step=1)
else:
    duration = st.sidebar.number_input("Duration (Hours)", min_value=3, step=3)  # Minimum 3 hours (1 module)

# Generate Curriculum button
if st.button("Generate Curriculum"):
    if not st.session_state.openai_api_key or not st.session_state.serper_api_key:
        st.warning("Please set up your API keys first.")
    elif course_name and duration:
        with st.spinner("Processing..."):
            json_file = "curriculum_data.json"
            user_query = f"{course_name} course for {duration} {duration_type.lower()}"

            output, timing_results = run_rag_pipeline(
                json_file, 
                user_query, 
                course_name=course_name, 
                duration=duration,
                duration_type=duration_type.lower()
            )

            if output:
                text_file = f'outputs/curriculum_{course_name.lower().replace(" ", "_")}.txt'
                pdf_file = f'outputs/curriculum_{course_name.lower().replace(" ", "_")}.pdf'


                cleaned_output = str(output).strip().replace("```", "").strip()


                # Save text content to file
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(str(cleaned_output))

                # Convert to PDF
                txt_to_pdf(text_file, pdf_file)

                st.success("Curriculum Generated!")
                
                # Display timing results in an expander
                with st.expander("View Processing Times"):
                    st.write("### Processing Times")
                    timing_df = pd.DataFrame({
                        'Step': timing_results.keys(),
                        'Time (seconds)': [f"{time:.2f}" for time in timing_results.values()]
                    })
                    st.dataframe(timing_df)
                    st.bar_chart(timing_df.set_index('Step')['Time (seconds)'].astype(float))

                # Display the plain text content
                with open(text_file, "r", encoding='utf-8') as f:
                    text_content = f.read()
                st.markdown(text_content)

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
