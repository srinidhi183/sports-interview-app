import pandas as pd
import numpy as np
import torch
import streamlit as st
import plotly.express as px
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import umap
import openai

# ---------- Streamlit Setup ----------
st.set_page_config(page_title="NLP Sports Interview Suite", layout="wide")
st.title("🏟️ Sports Interview Dashboard")

# ---------- OpenRouter API ----------
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"

# ---------- Caching ----------
@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained("Srinidhi186541564156/sports-interview-model")
    model = AutoModelForSequenceClassification.from_pretrained("Srinidhi186541564156/sports-interview-model")
    return tokenizer, model

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-mpnet-base-v2")

@st.cache_data
def load_training_data():
    return pd.read_csv("train.csv").dropna(subset=["Interview Text", "Labels"])

@st.cache_data
def load_generated_data():
    return pd.read_csv("Text_generation_output.csv")

# ---------- Load Resources ----------
tokenizer, clf_model = load_classifier()
embedder = load_embedder()
train_df = load_training_data()
gen_df = load_generated_data()

# ---------- Label Mapping ----------
unique_labels = sorted(train_df["Labels"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

tab1, tab2, tab3 = st.tabs(["📁 Transcript Classification", "🤖 AI Q&A", "📊 Clustering Visualization"])

# ---------- Tab 1: Transcript Classification ----------
with tab1:
    st.subheader("📁 Classify Full Interview Transcript")
    text = st.text_area("Enter full transcript text here:", height=180)
    if st.button("🔍 Predict Category"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = clf_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
        st.success(f"Predicted Category: **{id2label[pred_id]}**")

# ---------- Tab 2: Text Generation ----------
with tab2:
    st.subheader("🧠 Generate Interview Response")
    category = st.selectbox("Choose a category", options=label2id.keys())
    question = st.text_input("Enter your interview question")

    if st.button("🤖 Generate Response"):
        prompt = f"""You are an AI sports journalist assistant. Generate a professional answer.

Category: {category}
Question: "{question}"
Response:"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful sports assistant that generates realistic interview responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            output = response['choices'][0]['message']['content'].strip()
            st.success(output)
        except Exception as e:
            st.error(f"Error generating response: {e}")

# ---------- Tab 3: UMAP Clustering ----------
with tab3:
    st.subheader("📊 UMAP Clustering of Interview Texts")
    st.caption("Visualizing semantic structure of either training data or generated AI responses.")

    data_choice = st.radio("Choose data source:", ["Training Transcripts (train.csv)", "Generated Responses (Text_generation_output.csv)"])

    if st.button("🔄 Run Clustering"):
        if data_choice == "Training Transcripts (train.csv)":
            texts = train_df["Interview Text"].tolist()
            labels = train_df["Labels"].tolist()
        else:
            texts = gen_df["Response"].tolist()
            labels = gen_df["Category"].tolist()

        embeddings = embedder.encode(texts, show_progress_bar=True)
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=8, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced)

        label_map = {label: i for i, label in enumerate(sorted(set(labels)))}
        true_labels = [label_map[label] for label in labels]

        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)

        df_vis = pd.DataFrame(reduced, columns=["x", "y"])
        df_vis["Label"] = labels
        df_vis["Cluster"] = cluster_labels
        df_vis["Text"] = texts

        fig = px.scatter(
            df_vis, x="x", y="y", color="Label", hover_data=["Text"],
            title=f"UMAP Projection<br><sup>ARI={ari:.3f} | NMI={nmi:.3f}</sup>"
        )
        st.plotly_chart(fig, use_container_width=True)
