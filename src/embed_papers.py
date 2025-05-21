from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os

def embed_and_save(csv_path="data/arxiv_data.csv", model_name="all-MiniLM-L6-v2", output_path="model/arxiv_embeddings.pkl"):
    print("📥 Loading CSV...")
    df = pd.read_csv(csv_path)

    # Combine title + abstract
    texts = (df["title"] + ". " + df["abstract"]).tolist()

    print(f"🧠 Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"🔁 Embedding {len(texts)} documents...")
    embeddings = []
    for text in tqdm(texts, desc="Embedding"):
        emb = model.encode(text)
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Save both embeddings and metadata
    with open(output_path, "wb") as f:
        pickle.dump((embeddings, df), f)

    print(f"✅ Saved {len(embeddings)} embeddings to {output_path}")

if __name__ == "__main__":
    embed_and_save()