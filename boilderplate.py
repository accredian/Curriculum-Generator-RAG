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
