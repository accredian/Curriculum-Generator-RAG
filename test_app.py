from transformers import pipeline
import os
from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
from transformers import pipeline
from huggingface_hub import login
import json
os.environ['OPENAI_API_KEY'] = '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'
# os.environ['OPENAI_API_KEY'] = 'sk-cGNL7dFWnZchBHtALgJhT3BlbkFJ8eDktSCI6gwbeSew8D'

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
from datetime import datetime



# Add these imports at the top
import streamlit as st
from datetime import datetime
import os

# Add these caching functions
@st.cache_resource
def load_sentence_transformer(model_name='all-MiniLM-L6-v2'):
    """Load and cache the sentence transformer model"""
    return SentenceTransformer(model_name)

@st.cache_data
def load_json_with_timestamp(_json_file):
    """Load JSON data and return with last modified timestamp"""
    timestamp = os.path.getmtime(_json_file)
    with open(_json_file, "r") as file:
        return json.load(file), timestamp

@st.cache_data
def process_data_with_embeddings(data, timestamp):
    """Process data with embeddings, using timestamp for cache invalidation"""
    model = load_sentence_transformer()
    processed_data = []
    for record in data:
        embedding = model.encode(record["curriculum"]).tolist()
        processed_data.append({**record, "embedding": embedding})
    return processed_data

@st.cache_resource
class FAISSIndexManager:
    def __init__(self):
        self.index = None
        self.metadata = None
        self.last_updated = None
    
    def initialize_or_load(self, data, timestamp):
        """Initialize or load FAISS index based on data timestamp"""
        if self.last_updated != timestamp:
            dimension = len(data[0]["embedding"])
            self.index = faiss.IndexFlatL2(dimension)
            embeddings = np.array([record["embedding"] for record in data]).astype('float32')
            self.index.add(embeddings)
            self.metadata = {i: record for i, record in enumerate(data)}
            self.last_updated = timestamp
        return self.index, self.metadata

    def query(self, query_text, k=5):
        """Query the index using the cached sentence transformer"""
        model = load_sentence_transformer()
        query_embedding = model.encode(query_text).astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.metadata[idx] for idx in indices[0]]

#--------------------------------------------------------------------------------------------------------------------------------------



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
        3. Validate content quality and completeness, make sure each term has 4 modules
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






# Modify your run_rag_pipeline function
def run_rag_pipeline(json_file, user_query, course_name=None, duration=None, duration_type="months"):
    """
    Optimized RAG pipeline that caches intermediate results and only recomputes when necessary
    """
    # Load data with timestamp for cache invalidation
    data, timestamp = load_json_with_timestamp(json_file)
    
    # Process embeddings (cached unless data changes)
    processed_data = process_data_with_embeddings(data, timestamp)
    
    # Get or create FAISS index
    index_manager = FAISSIndexManager()
    index, metadata = index_manager.initialize_or_load(processed_data, timestamp)
    
    # Query the index
    results = index_manager.query(user_query)
    
    # Generate output
    return generate_llm_output(results, course_name=course_name, duration=duration, duration_type=duration_type)





def md_to_pdf(markdown_file, pdf_file):
    """
    Convert a Markdown file to a PDF file.
    :param markdown_file: Path to the input Markdown file.
    :param pdf_file: Path to the output PDF file.
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    html_content = markdown.markdown(markdown_content)

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


def sanitize_filename(course_name, max_length=50):
    """Create a safe, shortened filename from course name"""
    # Remove special characters and spaces
    safe_name = "".join(c for c in course_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_').lower()
    # Truncate if too long
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    return safe_name
# -------------------------------------------------------------------------------------------------------------

# Modify your Streamlit app code
st.title("Curriculum Generator")
st.sidebar.header("Please provide course details")

course_name = st.sidebar.text_input("Course Name")
duration_type = st.sidebar.radio("Select Duration Type", ["Months", "Hours"])

if duration_type == "Months":
    duration = st.sidebar.number_input("Duration (Months)", min_value=1, step=1)
else:
    duration = st.sidebar.number_input("Duration (Hours)", min_value=3, step=3)

# Initialize session state for tracking file changes
if 'last_file_update' not in st.session_state:
    st.session_state.last_file_update = None

if st.button("Generate Curriculum"):
    if course_name and duration:
        with st.spinner("Processing..."):
            json_file = "curriculum_data.json"
            
            # Create query based on duration type
            user_query = f"{course_name} course for {duration} {duration_type.lower()}"
            
            try:
                output = run_rag_pipeline(
                    json_file, 
                    user_query, 
                    course_name=course_name, 
                    duration=duration,
                    duration_type=duration_type.lower()
                )

                if output:
                    # Create shorter, sanitized filenames
                    safe_name = sanitize_filename(course_name)
                    markdown_file = f'outputs/curr_{safe_name}.md'
                    pdf_file = f'outputs/curr_{safe_name}.pdf'

                    # Ensure outputs directory exists
                    os.makedirs('outputs', exist_ok=True)

                    # Save markdown content
                    try:
                        with open(markdown_file, 'w', encoding='utf-8') as f:
                            f.write(str(output))

                        # Convert to PDF
                        with open(markdown_file, 'r', encoding='utf-8') as f:
                            markdown_content = f.read()
                        md_to_pdf(markdown_content, pdf_file)

                        st.success("Curriculum Generated!")
                        
                        # Display markdown
                        st.markdown("### Generated Curriculum (Markdown)")
                        st.markdown(str(output))

                        # Provide download button
                        with open(pdf_file, "rb") as f:
                            pdf_data = f.read()
                            st.download_button(
                                label="Download Curriculum as PDF",
                                data=pdf_data,
                                file_name=f"{safe_name}_curriculum.pdf",
                                mime="application/pdf"
                            )
                    except Exception as e:
                        st.error(f"File operation error: {str(e)}")
            except Exception as e:
                st.error(f"Error generating curriculum: {str(e)}")


            
    else:
        st.warning("Please provide all required inputs.")