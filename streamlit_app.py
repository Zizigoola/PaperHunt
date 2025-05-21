import streamlit as st
from src.search import PaperSearcher

from src.search import PaperSearcher


# Load model and index
@st.cache_resource
def load_searcher():
    return PaperSearcher()

searcher = load_searcher()

st.title("📚 Paper Hunt – Semantic Research Paper Finder")
st.markdown("Search for papers on Machine Learning, AI, or Deep Learning using semantic understanding.")

query = st.text_input("🔎 Enter your research topic:", placeholder="e.g., transformers for vision")

if query:
    results = searcher.search(query, top_k=5)
    st.write(f"### 🔍 Top {len(results)} Results for: `{query}`")
    for i, r in enumerate(results, 1):
        with st.expander(f"{i}. {r['title']}"):
            st.markdown(f"**📅 Date**: {r['update_date']}")
            st.markdown(f"**🏷️ Categories**: `{r['categories']}`")
            st.markdown(f"**🧾 Abstract:**\n\n{r['abstract']}")
