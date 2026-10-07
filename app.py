
from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF for PDF
import sqlite3  # Add this import
import json
from concurrent.futures import ThreadPoolExecutor
from logic import (
    calculate_match,
    recommend_jobs,
    recommend_courses,
    extract_skills_batch,
    extract_skills,
    find_similar_jobs,
)
from resume_match import find_similar_resumes

app = Flask(__name__)


import re

def extract_pdf_text(file_storage):
    try:
        file_storage.seek(0)
        doc = fitz.open(stream=file_storage.read(), filetype="pdf")
        text = " ".join([page.get_text() for page in doc])
        doc.close()
        
       
        prev = ""
        while text != prev:
            prev = text
            text = re.sub(r'(?<=\b[a-zA-Z]) (?=[a-zA-Z]\b)', '', text)
                
        return text.strip()
    except Exception as e:
        print(f"PDF Error: {e}")
        return ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match_resume():
    resume_text = ""
    job_desc = request.form.get('job_desc', '')

    if 'resumeFile' in request.files:
        file = request.files['resumeFile']
        if file.filename:
            if file.filename.lower().endswith('.pdf'):
                resume_text = extract_pdf_text(file)
            else:
                resume_text = file.read().decode('utf-8')

    if not resume_text:
        return jsonify({'error': 'No resume text detected. Please upload a valid file.'}), 400

    # OPTIMIZATION: Call Gemini ONCE for both texts to reduce latency by ~50%
    resume_skills, job_skills_extracted = extract_skills_batch(resume_text, job_desc)
    
    match_result = calculate_match(resume_skills, job_skills_extracted) if job_desc else {}
    
    # Store the result securely in the DB
    try:
        conn = sqlite3.connect("skillmatch.db")
        cursor = conn.cursor()
        
        # Used "User_Upload" as the source to distinguish it from the Kaggle dataset
        cursor.execute(
            """INSERT INTO resumes 
               (raw_text, source, matched_skills, missing_skills, extra_skills, job_desc_text) 
               VALUES (?, ?, ?, ?, ?, ?)""", 
            (
                resume_text, 
                "User_Upload",
                json.dumps(match_result.get('matched_skills', [])),
                json.dumps(list(set(match_result.get('missing_skills', [])))),
                json.dumps(match_result.get('extra_skills', [])),
                job_desc
            )
        )
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database Storage Error: {e}")
    similar_resumes = find_similar_resumes(resume_text, top_k=5)
    similar_jobs = find_similar_jobs(job_desc, top_k=5) if job_desc else []
    jobs = recommend_jobs(list(resume_skills))
    
    missing_skills = list(set(match_result.get('missing_skills', [])))
    courses = recommend_courses(missing_skills)

    return jsonify({
        'resume_skills': list(resume_skills),
        'match_score': match_result.get('score', 0),
        'matched_skills': match_result.get('matched_skills', []),
        'missing_skills': missing_skills,
        'extra_skills': match_result.get('extra_skills', []),
        'jobs': jobs,
        'courses': courses,
        'similar_resumes': similar_resumes,
        'similar_jobs': similar_jobs,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)