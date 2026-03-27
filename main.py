# ================================
# RESUME SCREENING SYSTEM
# ================================

# ---------- IMPORT LIBRARIES ----------
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')


# ---------- LOAD DATA ----------
resumes = pd.read_csv("Resume.csv")

print("Dataset Loaded Successfully\n")
print(resumes.head())
print("\nColumns:", resumes.columns)


# ---------- TEXT PREPROCESSING ----------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ✅ FIXED COLUMN NAME HERE
resumes['cleaned'] = resumes['Resume_str'].apply(preprocess)


# ---------- SKILL DATABASE ----------
skills_db = [
    "python", "java", "sql", "c++", "html", "css",
    "machine learning", "deep learning", "nlp",
    "data analysis", "excel", "power bi",
    "communication", "teamwork", "leadership"
]


# ---------- SKILL EXTRACTION ----------
def extract_skills(text):
    found = []
    for skill in skills_db:
        if skill in text:
            found.append(skill)
    return list(set(found))

resumes['skills'] = resumes['cleaned'].apply(extract_skills)


# ---------- JOB DESCRIPTION ----------
job_desc = """
We are looking for a Python developer with strong machine learning,
SQL, and data analysis skills. Knowledge of deep learning is a plus.
"""

clean_job_desc = preprocess(job_desc)
job_skills = extract_skills(clean_job_desc)

print("\nJob Skills Required:", job_skills)


# ---------- TF-IDF SIMILARITY ----------
vectorizer = TfidfVectorizer()

def get_similarity(resume, job):
    vectors = vectorizer.fit_transform([resume, job])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


# ---------- SKILL MATCH SCORE ----------
def skill_match_score(candidate_skills, job_skills):
    if len(job_skills) == 0:
        return 0
    match = len(set(candidate_skills) & set(job_skills))
    return match / len(job_skills)


# ---------- FINAL SCORING ----------
final_scores = []

for i, row in resumes.iterrows():
    sim_score = get_similarity(row['cleaned'], clean_job_desc)
    skill_score = skill_match_score(row['skills'], job_skills)
    
    # Weighted scoring
    final_score = (0.7 * sim_score) + (0.3 * skill_score)
    
    final_scores.append(final_score)

resumes['final_score'] = final_scores


# ---------- MISSING SKILLS ----------
def missing_skills(candidate_skills):
    return list(set(job_skills) - set(candidate_skills))

resumes['missing_skills'] = resumes['skills'].apply(missing_skills)


# ---------- RANKING ----------
ranked = resumes.sort_values(by='final_score', ascending=False)


# ---------- FINAL OUTPUT ----------
print("\n========== TOP CANDIDATES ==========")

top_n = 5

for i, row in ranked.head(top_n).iterrows():
    print("\n" + "="*50)
    print("Candidate ID:", row['ID'])
    print("Category:", row['Category'])
    print("Final Score:", round(row['final_score'], 3))
    print("Skills:", row['skills'])
    print("Missing Skills:", row['missing_skills'])


# ---------- VISUALIZATION (BONUS) ----------
top = ranked.head(10)

plt.figure()
plt.bar(range(len(top)), top['final_score'])
plt.xticks(range(len(top)), top['ID'], rotation=45)
plt.title("Top Candidates Ranking")
plt.xlabel("Candidate ID")
plt.ylabel("Score")
plt.tight_layout()
plt.show()