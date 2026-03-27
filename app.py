# ================================
# AI RESUME SCREENING SYSTEM
# ================================

import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from PyPDF2 import PdfReader

nltk.download('stopwords')


# ---------- CUSTOM UI STYLE ----------
st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #4CAF50;
}
.card {
    padding: 20px;
    border-radius: 10px;
    background-color: #1e1e1e;
    margin-bottom: 15px;
}
.skill-box {
    display: inline-block;
    padding: 6px 12px;
    margin: 5px;
    border-radius: 8px;
    background-color: #4CAF50;
    color: white;
}
.missing-box {
    display: inline-block;
    padding: 6px 12px;
    margin: 5px;
    border-radius: 8px;
    background-color: #FF4B4B;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD DATA ----------
df = pd.read_csv("Resume.csv")

# ---------- PREPROCESS ----------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned'] = df['Resume_str'].apply(preprocess)


# ---------- SKILLS ----------
skills_db = [
    "python", "java", "sql", "c++", "html", "css",
    "machine learning", "deep learning", "nlp",
    "data analysis", "excel", "power bi",
    "communication", "teamwork", "leadership"
]

def extract_skills(text):
    return [skill for skill in skills_db if skill in text]

df['skills'] = df['cleaned'].apply(extract_skills)


# ---------- MODEL ----------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['Category']

model = MultinomialNB()
model.fit(X, y)


# ---------- FILE READER ----------
def read_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    else:
        return uploaded_file.read().decode("utf-8")


# ---------- UI ----------
st.markdown('<div class="big-title">🚀 AI Resume Screening System</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    job_desc = st.text_area("📄 Job Description")

with col2:
    uploaded_file = st.file_uploader("📂 Upload Resume", type=["txt", "pdf"])


# ---------- PROCESS ----------
if uploaded_file is None:
    st.warning("Please upload a resume")

elif not job_desc.strip():
    st.warning("Please enter a job description")

else:
    resume_text = read_file(uploaded_file)

    clean_resume = preprocess(resume_text)
    clean_job = preprocess(job_desc)

    user_skills = extract_skills(clean_resume)
    job_skills = extract_skills(clean_job)

    # similarity
    vectors = vectorizer.transform([clean_resume, clean_job])
    sim_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    # skill match
    if len(job_skills) == 0:
        skill_score = 0
    else:
        skill_score = len(set(user_skills) & set(job_skills)) / len(job_skills)

    final_score = 0.7 * sim_score + 0.3 * skill_score

    # prediction
    role = model.predict(vectorizer.transform([clean_resume]))[0]

    # missing skills
    missing = list(set(job_skills) - set(user_skills))

    # ---------- RESULTS ----------
    st.subheader("📊 Results")

    # Role Card
    st.markdown(f"""
    <div class="card">
    <h3>🎯 Predicted Role</h3>
    <p style="color:#4CAF50; font-size:20px;">{role}</p>
    </div>
    """, unsafe_allow_html=True)

    # Score Card
    st.markdown(f"""
    <div class="card">
    <h3>📈 Match Score</h3>
    <p style="font-size:20px;">{round(final_score*100,2)}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(final_score * 100))

    # Skills
    st.subheader("🧠 Extracted Skills")
    for skill in user_skills:
        st.markdown(f'<span class="skill-box">{skill}</span>', unsafe_allow_html=True)

    # Missing Skills
    st.subheader("⚠️ Missing Skills")
    if missing:
        for skill in missing:
            st.markdown(f'<span class="missing-box">{skill}</span>', unsafe_allow_html=True)
    else:
        st.success("No missing skills 🎉")


# ---------- FOOTER ----------
st.markdown("---")
st.markdown("Developed by Vivek 🚀")