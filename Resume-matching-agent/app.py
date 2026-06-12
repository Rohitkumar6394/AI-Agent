import streamlit as st
import pandas as pd
import numpy as np

from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load Embedding Model
# -------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

model = load_model()

# -------------------------
# PDF Text Extraction
# -------------------------

def extract_text(pdf_file):

    text = ""

    reader = PdfReader(pdf_file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

# -------------------------
# UI
# -------------------------

st.title("AI Resume Screening Agent")

st.write(
    "Upload resumes and compare them with Job Description"
)

jd = st.text_area(
    "Enter Job Description",
    height=250
)

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

# -------------------------
# Process
# -------------------------

if st.button("Analyze Resumes"):

    if not uploaded_files:

        st.warning("Upload resumes first")
        st.stop()

    if not jd:

        st.warning("Enter Job Description")
        st.stop()

    jd_embedding = model.encode([jd])

    results = []

    for pdf in uploaded_files:

        resume_text = extract_text(pdf)

        resume_embedding = model.encode(
            [resume_text]
        )

        score = cosine_similarity(
            jd_embedding,
            resume_embedding
        )[0][0]

        score = round(score * 100, 2)

        status = (
            "Selected"
            if score >= 60
            else "Rejected"
        )

        results.append(
            {
                "Resume": pdf.name,
                "Score": score,
                "Status": status
            }
        )

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="Score",
        ascending=False
    )

    st.subheader("Results")

    st.dataframe(df)

    best_candidate = df.iloc[0]

    st.success(
        f"""
Best Candidate:
{best_candidate['Resume']}

Score:
{best_candidate['Score']}%
"""
    )

    df.to_csv(
        "result.csv",
        index=False
    )

    st.download_button(
        "Download Report",
        df.to_csv(index=False),
        file_name="result.csv",
        mime="text/csv"
    )