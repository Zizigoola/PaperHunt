import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class PaperSearcher:
    def __init__(self, embedding_path="model/arxiv_embeddings.pkl", model_name="all-MiniLM-L6-v2"):
        print("📦 Loading embeddings...")
        with open(embedding_path, "rb") as f:
            self.embeddings, self.df = pickle.load(f)

        self.model = SentenceTransformer(model_name)

        print("⚡ Initializing FAISS index...")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query, top_k=5):
        print(f"🔍 Searching for: {query}")
        query_vec = self.model.encode([query])
        D, I = self.index.search(query_vec, top_k)

        results = []
        for idx in I[0]:
            paper = self.df.iloc[idx]
            results.append({
                "title": paper["title"],
                "abstract": paper["abstract"][:500] + "...",
                "categories": paper["categories"],
                "update_date": paper["update_date"]
            })
        return results
