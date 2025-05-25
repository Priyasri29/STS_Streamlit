# hybrid_similarity_app_no_tf.py

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load SBERT once
@st.cache_resource
def load_sbert():
    return SentenceTransformer('all-MiniLM-L6-v2')

sbert_model = load_sbert()

# Setup TF-IDF and scaler on dummy data
@st.cache_resource
def setup_tfidf_and_scaler():
    dummy_texts = [
        "This is a sample sentence.",
        "Another example for TF-IDF vectorizer training.",
        "TF-IDF helps compute text similarity.",
        "SBERT gives embeddings."
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(dummy_texts)
    scaler = MinMaxScaler()
    scaler.fit([[0, 0], [1, 1]])
    return vectorizer, scaler

tfidf_vectorizer, scaler = setup_tfidf_and_scaler()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def row_cosine_sim(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]

def compute_similarity(text1, text2):
    t1 = clean_text(text1)
    t2 = clean_text(text2)

    sbert_vec1 = sbert_model.encode(t1)
    sbert_vec2 = sbert_model.encode(t2)

    tfidf_vec1 = tfidf_vectorizer.transform([t1])
    tfidf_vec2 = tfidf_vectorizer.transform([t2])

    sbert_sim = row_cosine_sim(sbert_vec1, sbert_vec2)
    tfidf_sim = row_cosine_sim(tfidf_vec1.toarray()[0], tfidf_vec2.toarray()[0])

    normalized = scaler.transform([[sbert_sim, tfidf_sim]])[0]
    norm_sbert, norm_tfidf = normalized

    w_sbert, w_tfidf = 0.8, 0.2
    hybrid_score = w_sbert * norm_sbert + w_tfidf * norm_tfidf

    return {
        'SBERT Similarity': sbert_sim,
        'TF-IDF Similarity': tfidf_sim,
        'Hybrid Weighted Similarity Score': hybrid_score
    }

st.title("Semantic Textual Similarity App")
st.markdown("Compute similarity using SBERT and TF-IDF.")

text1 = st.text_area("Enter Text 1:", height=150)
text2 = st.text_area("Enter Text 2:", height=150)

if st.button("Compute Similarity"):
    if text1.strip() == "" or text2.strip() == "":
        st.warning("Please enter text in both fields.")
    else:
        with st.spinner("Computing similarity..."):
            result = compute_similarity(text1, text2)
        st.success("Similarity scores calculated!")
        
        st.metric("Hybrid Weighted Similarity Score", f"{result['Hybrid Weighted Similarity Score']:.4f}")
        with st.expander("Show individual similarity scores"):
            st.write(f"**SBERT Similarity:** {result['SBERT Similarity']:.4f}")
            st.write(f"**TF-IDF Similarity:** {result['TF-IDF Similarity']:.4f}")
