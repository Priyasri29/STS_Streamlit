# -*- coding: utf-8 -*-
# hybrid_similarity_app.py

import pandas as pd
import numpy as np
import re
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ------------------------------
# 1. Load Models Once
# ------------------------------
@st.cache_resource
def load_models():
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return sbert, use

sbert_model, use_model = load_models()

# ------------------------------
# 2. Sample TF-IDF Setup using dummy data
# ------------------------------
@st.cache_resource
def setup_tfidf_and_scaler():
    dummy_texts = [
        "This is a sample sentence.",
        "Another example for TF-IDF vectorizer training.",
        "TF-IDF helps compute text similarity.",
        "SBERT and USE give embeddings."
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(dummy_texts)

    # Fake values to fit scaler range between [0,1]
    scaler = MinMaxScaler()
    scaler.fit([[0, 0, 0], [1, 1, 1]])
    return vectorizer, scaler

tfidf_vectorizer, scaler = setup_tfidf_and_scaler()

# ------------------------------
# 3. Preprocessing Function
# ------------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------
# 4. Cosine Similarity Function
# ------------------------------
def row_cosine_sim(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]

# ------------------------------
# 5. Hybrid Similarity Function
# ------------------------------
def compute_similarity(text1, text2):
    # Clean input texts
    t1 = clean_text(text1)
    t2 = clean_text(text2)

    # Generate embeddings
    sbert_vec1 = sbert_model.encode(t1)
    sbert_vec2 = sbert_model.encode(t2)

    use_vec1 = use_model([t1])[0].numpy()
    use_vec2 = use_model([t2])[0].numpy()

    # TF-IDF vectors
    tfidf_vec1 = tfidf_vectorizer.transform([t1])
    tfidf_vec2 = tfidf_vectorizer.transform([t2])

    # Individual cosine similarities
    sbert_sim = row_cosine_sim(sbert_vec1, sbert_vec2)
    use_sim = row_cosine_sim(use_vec1, use_vec2)
    tfidf_sim = row_cosine_sim(tfidf_vec1.toarray()[0], tfidf_vec2.toarray()[0])

    # Normalize the values
    normalized = scaler.transform([[sbert_sim, use_sim, tfidf_sim]])[0]
    norm_sbert, norm_use, norm_tfidf = normalized

    # Weighted score
    w_sbert, w_use, w_tfidf = 0.7, 0.2, 0.1
    hybrid_score = w_sbert * norm_sbert + w_use * norm_use + w_tfidf * norm_tfidf

    return {
        'SBERT Similarity': sbert_sim,
        'USE Similarity': use_sim,
        'TF-IDF Similarity': tfidf_sim,
        'Hybrid Weighted Similarity Score': hybrid_score
    }

# ------------------------------
# 6. Streamlit UI
# ------------------------------
st.title("Semantic Textual Similarity App")
st.markdown("Enter two texts below to compute how similar they are using a hybrid model combining SBERT, USE, and TF-IDF.")

# Input boxes
text1 = st.text_area("**Enter Text 1:**", height=150)
text2 = st.text_area("**Enter Text 2:**", height=150)

# Compute button
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
            st.write(f"**USE Similarity:** {result['USE Similarity']:.4f}")
            st.write(f"**TF-IDF Similarity:** {result['TF-IDF Similarity']:.4f}")
