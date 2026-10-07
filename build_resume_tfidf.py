import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib   # pip install joblib if needed

# 1. Load cleaned_text from DB
conn = sqlite3.connect("skillmatch.db")
df = pd.read_sql("SELECT resume_id, cleaned_text FROM resumes", conn)
conn.close()

corpus = df["cleaned_text"].fillna("").tolist()
resume_ids = df["resume_id"].tolist()

# 2. Fit TF-IDF
vectorizer = TfidfVectorizer(min_df=2)  # ignore very rare words
tfidf_matrix = vectorizer.fit_transform(corpus)

# 3. Save objects for later use in Flask
joblib.dump(vectorizer, "resume_tfidf_vectorizer.joblib")
joblib.dump(tfidf_matrix, "resume_tfidf_matrix.joblib")
joblib.dump(resume_ids, "resume_ids.joblib")

print("TF-IDF built for", len(resume_ids), "resumes")
