import sqlite3
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text

def build_job_tfidf():
    if not os.path.exists("data"):
        os.makedirs("data")

    conn = sqlite3.connect("skillmatch.db")
    cursor = conn.cursor()
    
    # Fetch rows. We check description_raw as the primary source of truth
    cursor.execute("SELECT id, title, description_raw, description_clean FROM jobs")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("Error: Database still empty. Please run add_real_data.py first!")
        return

    job_ids = []
    texts = []

    print("Processing job descriptions for TF-IDF...")
    for row in rows:
        jid, title, raw, clean = row
        # Use clean if available, otherwise clean the raw text now
        final_text = clean if (clean and clean.strip()) else clean_text(raw if raw else title)
        
        if final_text.strip():
            job_ids.append(jid)
            texts.append(final_text)

    if not texts:
        print("Error: No valid text found to train the model.")
        return

    # Initialize Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    print(f"Training TF-IDF model on {len(texts)} job entries...")
    matrix = vectorizer.fit_transform(texts)
    
    # Saving models
    joblib.dump(vectorizer, "data/job_tfidf_vectorizer.joblib")
    joblib.dump({"job_ids": job_ids, "matrix": matrix}, "data/job_tfidf_matrix.joblib")
    
    print(f"Successfully built TF-IDF matrix for {len(texts)} jobs.")

if __name__ == "__main__":
    build_job_tfidf()