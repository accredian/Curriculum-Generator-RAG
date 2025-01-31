# def generate_llm_output(results, max_new_tokens=200, course_name=None, duration=None, max_length=None):
#     """
#     Generate output from LLM using retrieved results.
    
#     :param results: List of matching curriculum records.
#     :param max_new_tokens: Maximum number of new tokens to generate.
#     :param course_name: Name of the course.
#     :param duration: Duration of the course in months.
#     :param max_length: Total maximum length for input and generated text.
#     :return: Generated text from LLM.
#     """
#     from transformers import pipeline
    
#     # Use the correct model identifier for Llama 3
#     model_id = "meta-llama/Llama-3.2-1B"
    
#     generator = pipeline(
#         "text-generation",
#         model=model_id,
#         torch_dtype="auto",
#         device_map="auto"
#     )
    
#     if not results or duration <= 0:
#         return "Invalid input data."

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
        
#         Ensure your output is distinct, clear, and aligned with the course's focus areas while adhering to the calculated structure.
#     """
    
#     # Calculate the appropriate max_length if not provided
#     if max_length is None:
#         max_length = len(prompt.split()) + max_new_tokens
    
#     output = generator(
#         prompt,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.95,
#         num_return_sequences=1,
#         max_length=max_length
#     )
    
#     return output[0]['generated_text'].strip()

# ---------------------- crewai code ----------------------------
    # Create specialized agents (previous agent definitions remain the same)
    # market_researcher = Agent(
    #     role='Market Research Specialist',
    #     goal='Research current industry trends and requirements',
    #     backstory='Expert in industry analysis with deep understanding of market demands',
    #     verbose=False,
    #     allow_delegation=True,
    #     tools=[tool]
    # )

    # curriculum_designer = Agent(
    #     role='Curriculum Architect',
    #     goal='Design curriculum following exact term-module-topic structure the curriculum that you design should ({num_terms} terms, {num_modules} modules, {num_topics} topics)',
    #     backstory='Senior curriculum designer specializing in structured learning paths',
    #     verbose=False,
    #     allow_delegation=True,
    #     tools=[tool]
    # )

    # content_enricher = Agent(
    #     role='Content Enhancement Specialist',
    #     goal='Add practical examples while maintaining structure',
    #     backstory='Expert in combining theory with real-world applications',
    #     verbose=False,
    #     allow_delegation=True,
    #     tools=[tool]
    # )

    # quality_reviewer = Agent(
    #     role='Quality Assurance Specialist',
    #     goal='Ensure final curriculum is well-formatted, maintains exact structure, and meets quality standards ',
    #     backstory='Experienced in curriculum validation and quality control with expertise in clear formatting',
    #     verbose=False,
    #     allow_delegation=True,
    #     tools=[tool, file_writer_tool]  # Add file_writer_tool to the quality reviewer
    # )

    # # Modified tasks with file saving
    # research_task = Task(
    #     description=f"""
    #     Research current trends and requirements for {course_name}:
    #     1. Identify industry trends and demands
    #     2. Research tools and technologies
    #     3. Find relevant case studies

    #     {base_prompt}
    #     """,
    #     expected_output="A comprehensive report of current industry trends, required skills, and market demands for the course topic",
    #     agent=market_researcher
    # )

    # design_task = Task(
    #     description=f"""
    #     Design the curriculum structure using research findings:
    #     1. Follow the exact term-module-topic structure
    #     2. Define learning objectives
    #     3. Ensure progression logic
    #     4. the curriculum that you design should ({num_terms} terms, {num_modules} modules, {num_topics} topics)

    #     {base_prompt}

        
    #     """,
    #     expected_output="A structured curriculum outline following the specified term-module-topic format with clear learning objectives",
    #     agent=curriculum_designer
    # )

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

    # # Modified review task to include file saving
    # review_task = Task(
    #     description=f"""
    #     Review, validate, format the final curriculum, and save to file:
    #     1. Verify exact structure compliance ({num_terms} terms, {num_modules} modules, {num_topics} topics)
    #     2. Ensure clear formatting and organization
    #     3. Validate content quality and completeness, make sure each term has 4 modules
    #     4. Format the final output in Markdown
    #     5. Save the curriculum to 'outputs/curriculum_{course_name.lower().replace(" ", "_")}.md'

    #     The final output must be perfectly formatted and ready for direct use.

    #     {base_prompt}
    #     """,
    #     expected_output="A final, perfectly formatted curriculum saved as a Markdown file",
    #     agent=quality_reviewer,
    #     output_file=f'outputs/curriculum_{course_name.lower().replace(" ", "_")}.md',
    #     create_directory=True
    # )

    # # Create and run the crew
    # crew = Crew(
    #     agents=[market_researcher, curriculum_designer, content_enricher, quality_reviewer],
    #     tasks=[research_task, design_task, enrich_task, review_task],
    #     memory=True,
    #     cache=True,
    #     max_rpm=100,
    #     share_crew=True,
    #     process=Process.sequential
    # )









