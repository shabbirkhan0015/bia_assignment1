"""
Offline RAG with MiniLM + FAISS
Optimized for ~4GB RAM laptops
"""

import os
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.manifold import TSNE
import joblib

# ---------------------------
# Utilities
# ---------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n\n".join(text)

def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 40) -> List[str]:
    """Split text into overlapping chunks (smaller for low RAM)."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load embedding model."""
    return SentenceTransformer(model_name)

def build_faiss_index(dim: int):
    """Create empty FAISS index."""
    return faiss.IndexFlatIP(dim)

def add_embeddings_to_index(index, embeddings: np.ndarray):
    """Add embeddings to FAISS in normalized batches."""
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

def get_top_k(index, query_emb: np.ndarray, k: int = 5):
    """Retrieve top-k results from FAISS index."""
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k)
    return D, I

# ---------------------------
# Main Pipeline
# ---------------------------

def process_file(file_path: str, persist_dir: str = "./rag_store", max_chunks=1500):
    """Load → chunk → embed → build FAISS (optimized for low RAM)."""
    os.makedirs(persist_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(file_path))[0]
    emb_path = os.path.join(persist_dir, f"{base}_emb.npy")
    idx_path = os.path.join(persist_dir, f"{base}_index.faiss")
    chunks_path = os.path.join(persist_dir, f"{base}_chunks.pkl")

    if os.path.exists(emb_path) and os.path.exists(idx_path) and os.path.exists(chunks_path):
        print("🔄 Loading cached index + embeddings...")
        embeddings = np.load(emb_path, mmap_mode="r")
        index = faiss.read_index(idx_path)
        chunks = joblib.load(chunks_path)
    else:
        # Read text
        if file_path.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        print(f"✅ Loaded file: {file_path}, length: {len(raw_text)} characters")

        # Chunking
        chunks = chunk_text(raw_text, chunk_size=400, chunk_overlap=40)
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            print(f"⚠️ Truncated to {max_chunks} chunks for memory safety")
        print(f"✂️ Split into {len(chunks)} chunks")

        # Embeddings (very small batch for low RAM)
        embed_model = load_embedding_model()
        embeddings = embed_model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=4   # 🔹 very small batch for 4GB RAM
        )

        # FAISS index (incremental)
        index = build_faiss_index(embeddings.shape[1])
        add_embeddings_to_index(index, embeddings.copy())

        # Save cache
        np.save(emb_path, embeddings)
        faiss.write_index(index, idx_path)
        joblib.dump(chunks, chunks_path)

        print("💾 Saved embeddings + index")

    return chunks, embeddings, index

# ---------------------------
# Retrieval
# ---------------------------

def ask(question: str, embed_model, chunks, index, top_k=5):
    """Retrieve chunks for a question."""
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    D, I = get_top_k(index, q_emb, k=top_k)
    results = [chunks[i][:400] for i in I[0] if i >= 0]

    print("\n🔎 Retrieved Context:")
    for r in results:
        print("-", r[:150], "...")
    return "\n\n".join(results)

# ---------------------------
# Visualization
# ---------------------------

def visualize_embeddings(embeddings, method="tsne", max_points=150):
    """2D visualization of embeddings with strong downsampling."""
    n_samples = min(max_points, embeddings.shape[0])
    idx = np.random.choice(embeddings.shape[0], n_samples, replace=False)
    X = embeddings[idx]

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=20, n_iter=500)  # lighter settings
        X2 = reducer.fit_transform(X)
    else:
        reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.5)
        X2 = reducer.fit_transform(X)

    plt.figure(figsize=(5,5))
    plt.scatter(X2[:,0], X2[:,1], s=6)
    plt.title(f"Embedding Visualization ({method.upper()})")
    plt.show()

# ---------------------------
# Run Example
# ---------------------------

if __name__ == "__main__":
    # Replace with your file path
    file_path = "abc.txt"  # or "example.txt"

    chunks, embeddings, index = process_file(file_path)

    # Load model once for questions
    embed_model = load_embedding_model()

    # Ask a question
    context = ask("What is the main contribution of the paper?", embed_model, chunks, index, top_k=3)

    # Visualize embeddings
    visualize_embeddings(embeddings, method="tsne", max_points=120)
