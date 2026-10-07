from dotenv import load_dotenv
load_dotenv()
import os
import re
import sqlite3
import json
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import clean_text
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


try:
    from google import genai
    from google.genai import types
    API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=API_KEY) if API_KEY else None
except ImportError:
    client = None

# Paths for data storage
DB_PATH = "skillmatch.db"
VECTORIZER_PATH = "data/job_tfidf_vectorizer.joblib"
MATRIX_PATH = "data/job_tfidf_matrix.joblib"

from skills_data import SKILL_LIST, CANONICAL_SKILLS

def normalize(text):
    """Standardizes text for skill extraction."""
    if not text: return ""
    # Remove special chars but keep '+' and '#' for C++, C#
    return re.sub(r'[^a-zA-Z0-9+#\s]', ' ', text.lower())

def remove_pii(text):
    """Strips emails and phone numbers to protect candidate privacy before sending to API."""
    if not text: return ""
    # Remove emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
    # Remove phone numbers
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]', text)
    # Remove common URLs (LinkedIn/Github)
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    return text

def extract_skills_batch(text_a, text_b):
    """
    Extracts skills from two texts simultaneously to reduce API latency by 50%.
    Returns a tuple: (resume_skills_set, job_skills_set)
    """
    res_set, job_set = set(), set()
    if not text_a and not text_b: return res_set, job_set
    
    safe_text_a = remove_pii(text_a or "")
    safe_text_b = remove_pii(text_b or "")

    # --- 1. Combined LLM Extraction ---
    if client is not None:
        try:
            prompt = (
                "Analyze these TWO professional texts and extract all professional hard skills and technical abilities. "
                "Return a JSON object with two keys: 'resume' and 'job', where each is a unique array of lowercase strings.\n\n"
                f"TEXT A (Resume): {safe_text_a[:3000]}\n\n"
                f"TEXT B (Job Description): {safe_text_b[:3000]}"
            )
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            data = json.loads(response.text)
            if 'resume' in data:
                for s in data['resume']: res_set.add(str(s).strip().lower())
            if 'job' in data:
                for s in data['job']: job_set.add(str(s).strip().lower())
            return res_set, job_set
        except Exception as e:
            print(f"Gemini Batch Extraction Error: {e}")

    # --- 2. Fallback (If LLM fails, run serial offline extraction) ---
    res_set = extract_skills(text_a) if text_a else set()
    job_set = extract_skills(text_b) if text_b else set()
    return res_set, job_set

def extract_skills(text):
    """
    1. Online Cloud AI Extraction using Google Gemini 2.5 Flash for perfect mapping.
    2. Fallback to Basic Hardcoded IT mapping if LLM is unavailable.
    """
    if not text: return set()
    found = set()
    
    # Clean PII out completely
    safe_text = remove_pii(text)
    
    # --- 1. LLM Online Smart Extraction ---
    if client is not None:
        try:
            # We truncate to 3500 chars (instead of 5000) for a smaller payload to improve API speed.
            prompt = f"Analyze the following text and extract all professional hard skills, core competencies, and technical abilities. Do not extract job titles (like 'Data Scientist' or 'Manager'), company names, locations, schools, soft skills, or adjectives. Return the response PURELY as a raw JSON array of lowercase strings with no markdown formatting or extra text.\nText: {safe_text[:3500]}"
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            # Parse the returned JSON array
            extracted_list = json.loads(response.text)
            for skill in extracted_list:
                found.add(str(skill).strip().lower())
            return found
        except Exception as e:
            print(f"Gemini LLM Error: {e}")
            # If LLM fails, fall through to fallback
            
    # --- 2. Offline Fallback (IT Hardcoded) ---
    norm_text = normalize(text)
    
    for skill in SKILL_LIST:
        if " " in skill and skill in norm_text:
            found.add(skill)
            
    for skill in SKILL_LIST:
        if " " not in skill:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, norm_text):
                found.add(skill)
                
    return found

def get_canonical_name(skill):
    """Maps skill variations to professional display names."""
    return CANONICAL_SKILLS.get(skill.lower(), skill.title())

def calculate_match(resume_raw, job_raw):
    """Computes technical skill overlap and generates match score using pre-extracted skills to save time."""
    resume_skills = {get_canonical_name(s) for s in resume_raw}
    job_skills = {get_canonical_name(s) for s in job_raw}
    
    matched = resume_skills & job_skills
    
    # --- Advanced Semantic Matching (Porter Stemming) ---
    # This ensures "Illustration" and "Illustrator" are identified as the same root.
    remaining_res = resume_skills - matched
    remaining_job = job_skills - matched
    
    if remaining_res and remaining_job:
        # Create a mapping of stems to original skill names for the resume
        res_stems = {}
        for r_skill in remaining_res:
            # We stem each word in the skill (e.g. "Graphic Design" -> "graphic design")
            # But for simplicity and speed, we stem the whole string if it's single word
            s = stemmer.stem(r_skill.lower())
            res_stems[s] = r_skill
            
        for j_skill in remaining_job:
            j_stem = stemmer.stem(j_skill.lower())
            if j_stem in res_stems:
                matched.add(j_skill) # Add the job's version to matched
                # We don't remove from missing here, the set difference handles it below
    
    missing = job_skills - matched
    extra = resume_skills - matched
    
    total_req = len(job_skills)
    # Score calculation: how many required skills does the candidate have?
    score = round((len(matched) / total_req) * 100, 2) if total_req > 0 else 0
    
    return {
        "resume_skills": sorted(list(resume_skills)),
        "job_skills": sorted(list(job_skills)),
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing)),
        "extra_skills": sorted(list(extra)),
        "score": min(score, 100.0)
    }

import urllib.parse

def recommend_jobs(resume_skills, top_k: int = 5):
    """Generates real-time LinkedIn search links based on resume skills."""
    if not resume_skills:
        return []

    query_text = " ".join(sorted(resume_skills))
    
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        data = joblib.load(MATRIX_PATH)
        job_ids, X_jobs = data["job_ids"], data["matrix"]
        
        x_query = vectorizer.transform([query_text])
        sims = cosine_similarity(x_query, X_jobs)[0]
        
        # 1. First, attempt to use the ML TF-IDF Engine
        # Set a stricter 33% threshold (0.33) to filter out "false positives" or outliers.
        top_indices = [i for i in np.argsort(sims)[::-1] if sims[i] > 0.33][:top_k]

        results = []
        
        # 2. If the ML Engine found related jobs, use them!
        if top_indices:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in top_indices)
            top_job_ids = [job_ids[i] for i in top_indices]
            cur.execute(f"SELECT id, title, description_raw FROM jobs WHERE id IN ({placeholders})", top_job_ids)
            rows = {r[0]: r for r in cur.fetchall()}
            conn.close()

            for idx in top_indices:
                jid = job_ids[idx]
                if jid in rows:
                    row = rows[jid]
                    raw_title = row[1]
                    clean_title = re.split(r'[-(\[|]', raw_title)[0].strip()
                    safe_query = urllib.parse.quote_plus(clean_title)
                    real_world_url = f"https://www.linkedin.com/jobs/search/?keywords={safe_query}"

                    results.append({
                        "job_id": row[0],
                        "title": raw_title,
                        "description": row[2][:150] + "..." if row[2] else "Matching job role...",
                        "url": real_world_url,
                        "score": round(float(sims[idx]) * 100, 1)
                    })
            return results

        # 3. DYNAMIC HEURISTIC FALLBACK (If the database is missing their career field)
        # Instead of generic heuristics, we ask the LLM to generate 3 perfect real-world job titles for their skillset.
        else:
            fallback_skills = sorted(list(resume_skills))[:6] # Send top 6 skills for context
            job_titles = []
            
            if client is not None and fallback_skills:
                try:
                    prompt = f"Given these professional skills: {', '.join(fallback_skills)}. Generate exactly 3 very common, real-world job titles (like 'Graphic Designer', 'Financial Analyst', 'Registered Nurse', 'Mechanical Engineer') that perfectly match these skills. Return the response PURELY as a raw JSON array of strings with no markdown formatting."
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json"),
                    )
                    titles = json.loads(response.text)
                    if isinstance(titles, list) and len(titles) > 0:
                        job_titles = [str(t).title() for t in titles[:3]]
                except Exception as e:
                    print(f"Gemini fallback title error: {e}")
            
            # If LLM fails or is unavailable, use generic fallback
            if not job_titles:
                for skill in fallback_skills[:3]:
                    skill_lower = skill.lower()
                    if skill_lower in ("drawing", "painting", "sketching", "illustration", "visual arts", "fine arts", "charcoal", "art", "pencil art"):
                        job_titles.append(f"{skill.title()} Artist")
                    elif "js" in skill_lower or "design" in skill_lower or "developer" in skill_lower:
                        job_titles.append(f"{skill.title()} Professional")
                    else:
                        job_titles.append(f"{skill.title()} Specialist")

            import random
            for job_title in job_titles:
                safe_query = urllib.parse.quote_plus(job_title)
                results.append({
                    "job_id": random.randint(9000, 9999),
                    "title": job_title,
                    "description": f"Dynamic real-time recommendation customized perfectly for your skill profile. Click to search live jobs!",
                    "url": f"https://www.linkedin.com/jobs/search/?keywords={safe_query}",
                    "score": "Live"
                })
            return results
    except Exception as e:
        print(f"Recommend Jobs Error: {e}")
        return []

def recommend_courses(missing_skills):
    """Generates Coursera search links for skills the user is missing."""
    if not missing_skills:
        return []
    return [
        {
            "skill": s, 
            "course_name": f"{s.title()} Professional Certification", 
            "provider": "Coursera",
            "url": f"https://www.coursera.org/search?query={s.replace(' ', '%20')}"
        } for s in sorted(missing_skills)
    ]

def find_similar_jobs(job_text: str, top_k: int = 5):
    """Finds jobs similar to the provided Job Description."""
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        data = joblib.load(MATRIX_PATH)
        job_ids, X_jobs = data["job_ids"], data["matrix"]
        
        cleaned = clean_text(job_text or "")
        x_query = vectorizer.transform([cleaned])
        sims = cosine_similarity(x_query, X_jobs)[0]
        
        top_indices = [i for i in np.argsort(sims)[::-1] if sims[i] > 0.15][:top_k]
        top_job_ids = [job_ids[i] for i in top_indices]
        
        if not top_job_ids:
            return []

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        placeholders = ",".join("?" for _ in top_job_ids)
        cur.execute(f"SELECT id, title, description_raw, url FROM jobs WHERE id IN ({placeholders})", top_job_ids)
        rows = {r[0]: r for r in cur.fetchall()}
        conn.close()

        results = []
        for idx, jid in zip(top_indices, top_job_ids):
            # Safety check ensures no crash if database row is missing
            if jid in rows:
                title = rows[jid][1]
                results.append({
                    "job_id": jid, 
                    "title": title, 
                    "description": rows[jid][2][:150] + "..." if rows[jid][2] else "Role details...", 
                    "url": rows[jid][3] if rows[jid][3] else f"https://www.linkedin.com/jobs/search/?keywords={title.replace(' ', '+')}",
                    "score": round(float(sims[idx]) * 100, 1)
                })
        return results
    except Exception as e:
        print(f"Find Similar Jobs Error: {e}")
        return []