'''
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

# ------------ STREAMLIT UI ------------
st.set_page_config(page_title="NLP Sports Interview Suite", layout="wide")
st.title("🏟️ Sports Interview Dashboard")

# ------------ API KEY ------------
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"

# ------------ CACHING ------------
@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("RoBERTa-base")
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

tokenizer, clf_model = load_classifier()
embedder = load_embedder()
train_df = load_training_data()
gen_df = load_generated_data()

# ------------ LABEL MAPS ------------
unique_labels = sorted(train_df["Labels"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}



tab1, tab2, tab3 = st.tabs(["📁 Transcript Classification", "🤖 AI Q&A", "📊 Clustering Visualization"])

# ------------ TAB 1: TRANSCRIPT CLASSIFICATION ------------
with tab1:
    st.subheader("📁 Classify Full Interview Transcript")
    text = st.text_area("Enter full transcript text here:", height=180)
    if st.button("🔍 Predict Category"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = clf_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
        st.success(f"Predicted Category: **{id2label[pred_id]}**")

# ------------ TAB 2: TEXT GENERATION ------------
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

# ------------ TAB 3: UMAP CLUSTERING OF AI RESPONSES ------------
with tab3:
    st.subheader("📊 UMAP Clustering of Generated Responses")
    st.caption("Visualizing AI-generated interview responses based on semantic similarity.")

    if st.button("🔄 Run Clustering"):
        embeddings = embedder.encode(gen_df["Response"].tolist(), show_progress_bar=True)
        umap_model = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
        reduced = umap_model.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=8, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced)

        label_map = {
            "Game Strategy": 0,
            "Player Performance": 1,
            "Injury Updates": 2,
            "Post-Game Analysis": 3,
            "Team Morale": 4,
            "Upcoming Matches": 5,
            "Off-Game Matters": 6,
            "Controversies": 7
        }
        true_labels = [label_map[cat] for cat in gen_df["Category"]]

        # Score
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)

        df_vis = pd.DataFrame(reduced, columns=["x", "y"])
        df_vis["True Label"] = gen_df["Category"]
        df_vis["Cluster"] = cluster_labels
        df_vis["Response"] = gen_df["Response"].str[:150]

        fig = px.scatter(
            df_vis, x="x", y="y", color="True Label", hover_data=["Response"],
            title=f"UMAP Projection of Interview Responses<br><sup>ARI={ari:.3f} | NMI={nmi:.3f}</sup>"
        )
        st.plotly_chart(fig, use_container_width=True)


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

# ------------ STREAMLIT UI ------------
st.set_page_config(page_title="NLP Sports Interview Suite", layout="wide")
st.title("🏟️ Sports Interview Dashboard")

# ------------ API KEY ------------
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"

# ------------ CACHING ------------
@st.cache_resource
def load_classifier():
    # Load the fine-tuned model and tokenizer
    model_path = "./best_model"  # Path to your fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
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

tokenizer, clf_model = load_classifier()
embedder = load_embedder()
train_df = load_training_data()
gen_df = load_generated_data()

# ------------ LABEL MAPS ------------
unique_labels = sorted(train_df["Labels"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

tab1, tab2, tab3 = st.tabs(["📁 Transcript Classification", "🤖 AI Q&A", "📊 Clustering Visualization"])

# ------------ TAB 1: TRANSCRIPT CLASSIFICATION ------------
with tab1:
    st.subheader("📁 Classify Full Interview Transcript")
    text = st.text_area("Enter full transcript text here:", height=180)
    if st.button("🔍 Predict Category"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = clf_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
        st.success(f"Predicted Category: **{id2label[pred_id]}**")

# ------------ TAB 2: TEXT GENERATION ------------
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

# ------------ TAB 3: UMAP CLUSTERING OF AI RESPONSES ------------
with tab3:
    st.subheader("📊 UMAP Clustering of Generated Responses")
    st.caption("Visualizing AI-generated interview responses based on semantic similarity.")

    if st.button("🔄 Run Clustering"):
        embeddings = embedder.encode(gen_df["Response"].tolist(), show_progress_bar=True)
        umap_model = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
        reduced = umap_model.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=8, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced)

        label_map = {
            "Game Strategy": 0,
            "Player Performance": 1,
            "Injury Updates": 2,
            "Post-Game Analysis": 3,
            "Team Morale": 4,
            "Upcoming Matches": 5,
            "Off-Game Matters": 6,
            "Controversies": 7
        }
        true_labels = [label_map[cat] for cat in gen_df["Category"]]

        # Score
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)

        df_vis = pd.DataFrame(reduced, columns=["x", "y"])
        df_vis["True Label"] = gen_df["Category"]
        df_vis["Cluster"] = cluster_labels
        df_vis["Response"] = gen_df["Response"].str[:150]

        fig = px.scatter(
            df_vis, x="x", y="y", color="True Label", hover_data=["Response"],
            title=f"UMAP Projection of Interview Responses<br><sup>ARI={ari:.3f} | NMI={nmi:.3f}</sup>"
        )
        st.plotly_chart(fig, use_container_width=True)
'''

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
openai.api_key = "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"  # Replace with your actual key
openai.api_base = "https://openrouter.ai/api/v1"

# ---------- Caching ----------
@st.cache_resource
def load_classifier():
    model_path = "./best_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
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

# ---------- Load Models & Data ----------
tokenizer, clf_model = load_classifier()
embedder = load_embedder()
train_df = load_training_data()
gen_df = load_generated_data()

# ---------- Label Mapping ----------
unique_labels = sorted(train_df["Labels"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

tab1, tab2, tab3 = st.tabs(["📁 Transcript Classification", "🤖 AI Q&A", "📊 Clustering Visualization"])

# ---------- Tab 1: Classification ----------
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

# ---------- Tab 3: Clustering Visualization ----------
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

        # Score if true labels exist
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
