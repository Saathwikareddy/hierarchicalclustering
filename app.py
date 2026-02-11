import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import linkage, dendrogram

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    page_icon="🟣",
    layout="wide"
)

st.title("🟣 News Topic Discovery Dashboard")
st.write(
    "This system uses **Hierarchical Clustering** to automatically group "
    "similar news articles based on textual similarity."
)

st.markdown(
    "👉 *Discover hidden themes without defining categories upfront.*"
)

# ---------------------------------------------------
# SIDEBAR – INPUT CONTROLS
# ---------------------------------------------------
st.sidebar.header("📂 Dataset & Clustering Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

@st.cache_data
def load_default_data():
    df = pd.read_csv("/mnt/data/all-data.csv", encoding="latin1", header=None)
    df.columns = ["label", "text"]
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()

# Auto-detect text column
text_column = None
for col in df.columns:
    if df[col].dtype == "object":
        text_column = col
        break

texts = df[text_column].astype(str)

# -------------------------------
# TEXT VECTORIZATION CONTROLS
# -------------------------------
st.sidebar.subheader("📝 Text Vectorization")

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features",
    100, 2000, 1000
)

use_stopwords = st.sidebar.checkbox(
    "Use English Stopwords",
    value=True
)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

# -------------------------------
# HIERARCHICAL CONTROLS
# -------------------------------
st.sidebar.subheader("🌳 Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

distance_metric = st.sidebar.selectbox(
    "Distance Metric",
    ["euclidean"]
)

dendro_size = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    20, 200, 100
)

# ---------------------------------------------------
# TF-IDF + SVD
# ---------------------------------------------------
@st.cache_data
def vectorize_text(texts):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english" if use_stopwords else None,
        ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

X_tfidf, tfidf = vectorize_text(texts)

@st.cache_data
def reduce_dimensions(X):
    svd = TruncatedSVD(n_components=100, random_state=42)
    return svd.fit_transform(X)

X_reduced = reduce_dimensions(X_tfidf)

# ---------------------------------------------------
# DENDROGRAM SECTION
# ---------------------------------------------------
st.header("🌳 Dendrogram Construction & Analysis")

if st.button("🟦 Generate Dendrogram"):
    X_sub = X_reduced[:dendro_size]

    Z = linkage(X_sub, method=linkage_method)

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, no_labels=True)
    ax.set_ylabel("Distance")
    ax.set_xlabel("Article Index")
    ax.set_title("Hierarchical Clustering Dendrogram")

    st.pyplot(fig)

    st.info(
        "🔍 Look for large vertical gaps. These indicate strong topic separation. "
        "Choose cluster count by cutting before large jumps."
    )

# ---------------------------------------------------
# APPLY CLUSTERING
# ---------------------------------------------------
st.header("🔗 Apply Hierarchical Clustering")

n_clusters = st.slider(
    "Select Number of Clusters",
    2, 15, 8
)

if st.button("🟩 Apply Clustering"):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    )

    clusters = model.fit_predict(X_reduced)
    df["cluster"] = clusters

    # ---------------------------------------------------
    # PCA VISUALIZATION
    # ---------------------------------------------------
    st.subheader("📈 Cluster Visualization (2D Projection)")

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_reduced)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    scatter = ax2.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=clusters,
        cmap="tab10",
        alpha=0.6
    )
    ax2.set_title("2D Projection of News Clusters")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")
    st.pyplot(fig2)

    # ---------------------------------------------------
    # SILHOUETTE SCORE
    # ---------------------------------------------------
    st.subheader("📊 Validation (Without Labels)")

    sil_score = silhouette_score(X_reduced, clusters)

    st.metric("Silhouette Score", round(sil_score, 3))

    st.caption(
        "Close to 1 → well-separated clusters | "
        "Close to 0 → overlapping | "
        "Negative → poor clustering"
    )

    # ---------------------------------------------------
    # CLUSTER SUMMARY (BUSINESS VIEW)
    # ---------------------------------------------------
    st.subheader("📋 Cluster Summary")

    terms = tfidf.get_feature_names_out()

    summary = []

    for c in sorted(df["cluster"].unique()):
        idx = df[df["cluster"] == c].index
        mean_tfidf = X_tfidf[idx].mean(axis=0)
        top_terms = np.argsort(mean_tfidf.A1)[-10:]
        keywords = ", ".join(terms[top_terms])
        snippet = df.loc[idx[0], text_column][:150]

        summary.append([c, len(idx), keywords, snippet])

    summary_df = pd.DataFrame(
        summary,
        columns=[
            "Cluster ID",
            "Number of Articles",
            "Top Keywords",
            "Representative Article Snippet"
        ]
    )

    st.dataframe(summary_df, use_container_width=True)

    # ---------------------------------------------------
    # BUSINESS INTERPRETATION
    # ---------------------------------------------------
    st.subheader("🧠 Business Interpretation")

    for _, row in summary_df.iterrows():
        st.markdown(
            f"🟣 **Cluster {row['Cluster ID']}**: "
            f"Articles related to **{row['Top Keywords'].split(',')[0:3]}**"
        )

    # ---------------------------------------------------
    # USER GUIDANCE
    # ---------------------------------------------------
    st.info(
        "Articles grouped in the same cluster share similar vocabulary and themes. "
        "These clusters can be used for **automatic tagging**, "
        "**content recommendations**, and **editorial organization**."
    )
