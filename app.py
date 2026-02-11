import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🟣 News Topic Discovery Dashboard")
st.write(
    "This system uses **Hierarchical Clustering** to automatically group "
    "similar news articles based on textual similarity."
)

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/news.csv")  # dataset inside repo

df = load_data()

st.success("Dataset loaded from repository")

# ---------------- TEXT COLUMN ----------------
st.sidebar.header("📝 Text Column")
text_column = st.sidebar.selectbox(
    "Select text column",
    df.columns
)

texts = df[text_column].astype(str)

# ---------------- TF-IDF CONTROLS ----------------
st.sidebar.header("📝 Text Vectorization")

max_features = st.sidebar.slider("Max TF-IDF Features", 100, 2000, 1000)
use_stopwords = st.sidebar.checkbox("Remove English Stopwords", value=True)

ngram_choice = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

ngram_range = (1, 1) if ngram_choice == "Unigrams" else (2, 2) if ngram_choice == "Bigrams" else (1, 2)

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if use_stopwords else None,
    ngram_range=ngram_range
)

X_tfidf = vectorizer.fit_transform(texts)

# ---------------- DENDROGRAM ----------------
st.sidebar.header("🌳 Hierarchical Controls")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

subset_size = st.sidebar.slider("Articles for Dendrogram", 20, 200, 50)

if st.sidebar.button("🟦 Generate Dendrogram"):
    st.subheader("🌳 Dendrogram (Subset)")

    X_subset = X_tfidf[:subset_size].toarray()
    Z = linkage(X_subset, method=linkage_method)

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax)
    ax.set_ylabel("Distance")
    ax.set_xlabel("Article Index")

    st.pyplot(fig)

# ---------------- APPLY CLUSTERING ----------------
st.sidebar.header("🟩 Apply Clustering")

num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

model = AgglomerativeClustering(
    n_clusters=num_clusters,
    linkage=linkage_method
)

clusters = model.fit_predict(X_tfidf.toarray())
df["Cluster"] = clusters

# ---------------- PCA VISUALIZATION ----------------
st.subheader("📉 Cluster Visualization (PCA)")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.7)
ax2.set_xlabel("PCA Component 1")
ax2.set_ylabel("PCA Component 2")

st.pyplot(fig2)

# ---------------- SILHOUETTE SCORE ----------------
st.subheader("📊 Clustering Validation")

score = silhouette_score(X_tfidf, clusters)
st.metric("Silhouette Score", round(score, 3))

# ---------------- BUSINESS INTERPRETATION ----------------
st.subheader("🧠 Business Interpretation")

for c in range(num_clusters):
    st.write(
        f"🟣 Cluster {c}: Articles grouped based on shared vocabulary and themes."
    )

# ---------------- USER GUIDANCE ----------------
st.info(
    "Articles in the same cluster share similar language patterns. "
    "These clusters help in automatic tagging, recommendations, "
    "and content organization."
)
