import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import json

load_dotenv()

os.getenv("GROQ_API_KEY")

class Chain:
    def __init__(self):
        self.llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"))
        
    def extract_jobs(self,cleaned_text):
        prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {cleaned_text}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """)

        chain_extract = prompt_extract | self.llm 
        res = chain_extract.invoke(input={'cleaned_text':cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]
        
    def extract_student_info(self,resume_text):
        prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM RESUME:
        {resume_text}
        ### INSTRUCTION:
        The scraped text is from the Resume of a student.
        Your job is to extract the tech skills, projects, internship experience or any other experiences,certifications or extra activities and return them in JSON format containing the 
        following keys: `r_skills`, `r_experience`, `r_projects` and `others`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """
        )
        chain_extract = prompt_extract | self.llm 
        res = chain_extract.invoke(input={'resume_text':resume_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]


    def write_mail(self, job, resume):
        """
        Generates two emails:
        - A general cold email without resume details.
        - A personalized email using resume details.
        """
        # Generic email (without resume context)
        prompt_generic = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are a candidate applying for the above job. Write a cold email to the recruiter 
            expressing interest and requesting an opportunity to interview.
            Do not reference any specific personal details.

            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_generic = prompt_generic | self.llm
        generic_email = chain_generic.invoke({"job_description": str(job)}).content

        # Personalized email (with resume context)
        prompt_personalized = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are a student at Manipal Institute Of Technology pursuing B.Tech in CSE(AIML) with a minor specialization in Advanced NLP.
            You are looking for off-campus opportunities and reaching out to recruiters.
            Write a cold email using the following details:
            - Experience: {resume_r_experience}
            - Projects: {resume_r_projects}

            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_personalized = prompt_personalized | self.llm
        personalized_email = chain_personalized.invoke({
            "job_description": str(job),
            "resume_r_experience": str(resume),
            "resume_r_projects": str(resume),
        }).content

        return generic_email, personalized_email



if __name__=="__main__":
    print(os.getenv("GROQ_API_KEY"))














