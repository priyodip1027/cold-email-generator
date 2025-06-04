# main.py
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from langchain.document_loaders import PyPDFLoader
import tempfile
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import time

def create_streamlit_app(llm):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a URL:")
    resume_file = st.file_uploader("Upload your resume", type="pdf")
    submit_button = st.button("Submit")

    if submit_button:
        if not url_input:
            st.error("Please enter a valid URL.")
            return

        try:
            # ChromaDB setup
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            client = chromadb.Client()
            resume_collection = client.get_or_create_collection("resumes", embedding_function=embedding_func)
            job_collection = client.get_or_create_collection("jobs", embedding_function=embedding_func)

            # Load and process job data
            loader = WebBaseLoader([url_input])
            page_data = loader.load().pop().page_content
            jobs = llm.extract_jobs(page_data)

            # Store job data
            for idx, job in enumerate(jobs):
                job_text = f"Role: {job.get('role')}\nSkills: {job.get('skills')}\nDescription: {job.get('description')}"
                job_collection.add(
                    documents=[job_text],
                    ids=[f"job_{idx}_{hash(job_text)}"],
                    metadatas=[{"type": "job", "source": "scraped"}]
                )

            if resume_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(resume_file.read())
                    temp_file_path = temp_file.name

                r_loader = PyPDFLoader(temp_file_path)
                documents = r_loader.load()
                pdf_text = "\n".join([doc.page_content for doc in documents])
                resume = llm.extract_student_info(pdf_text)

                # Store resume in Chroma
                resume_text = "\n".join([f"{k}: {v}" for k, v in resume[0].items()])
                resume_collection.add(
                    documents=[resume_text],
                    ids=[f"resume_{hash(resume_text)}"],
                    metadatas=[{"type": "resume", "source": "upload"}]
                )

                # Generate Emails
                generic_email, personalized_email = llm.write_mail(jobs, resume)

                # Display Emails
                st.subheader("Generic Cold Email (Without Resume)")
                st.code(generic_email, language="markdown")
                st.subheader("Personalized Cold Email (With Resume)")
                st.code(personalized_email, language="markdown")

                def extract_text(data):
                    if isinstance(data, list):
                        return ', '.join([str(item) for item in data])
                    return str(data)

                def evaluate_factualness(email, ground_truth):
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    gt_text = f"Skills: {extract_text(ground_truth['skills'])}\n" \
                               f"Projects: {extract_text(ground_truth['projects'])}\n" \
                               f"Experience: {extract_text(ground_truth['experience'])}\n" \
                               f"Job Requirements: {extract_text(ground_truth['job_requirements'])}"
                    email_embedding = model.encode(email)
                    ground_truth_embedding = model.encode(gt_text)
                    return round(cosine_similarity([email_embedding], [ground_truth_embedding])[0][0], 3)

                # Evaluate factualness
                ground_truth = {
                    "skills": resume[0]["r_skills"],
                    "projects": resume[0]["r_projects"],
                    "experience": resume[0]["r_experience"],
                    "job_requirements": jobs
                }

                generic_factualness = evaluate_factualness(generic_email, ground_truth)
                personalized_factualness = evaluate_factualness(personalized_email, ground_truth)

                # Skill Match Accuracy
                resume_skills = set(map(str.lower, resume[0]["r_skills"]))
                job_skills = set()
                for job in jobs:
                    job_skills.update(map(str.lower, job.get("skills", [])))
                matched_skills = resume_skills & job_skills
                skill_match_accuracy = round((len(matched_skills) / len(job_skills)) * 100, 2) if job_skills else 0

                # Additional Metrics
                personalization_score = round(personalized_factualness * 100, 2)
                generic_personalization = round(generic_factualness * 100, 2)
                hallucination_generic = round((1 - generic_factualness) * 100, 2)
                hallucination_personalized = round((1 - personalized_factualness) * 100, 2)
                hallucination_diff = hallucination_generic - hallucination_personalized

                # Table Output
                st.subheader("📊 Results Summary Table")
                results_data = [
                    ["Factualness Score (0–1)", generic_factualness, personalized_factualness, f"+{round(((personalized_factualness - generic_factualness) / generic_factualness) * 100, 2)}%"],
                    ["Personalization Score (0–100)", generic_personalization, personalization_score, f"+{personalization_score - generic_personalization}%"],
                    ["Hallucination Rate (%)", f"{hallucination_generic}%", f"{hallucination_personalized}%", f"-{hallucination_diff}%"],
                    ["Skill Match Accuracy (%)", "-", f"{skill_match_accuracy}%", "-"]
                ]
                st.table(pd.DataFrame(results_data, columns=["Metric", "Generic", "With Resume + FEWL", "Improvement"]))

        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain)
