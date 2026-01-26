import os
import re
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_to_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def build_system_kb_store_all_columns(
    excel_path: str,
    save_dir: str = "system_kb_store",
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    embed_column: str = "Description of Contract",
):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n Loading Excel Knowledge Base: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f" Loaded rows={len(df)} cols={len(df.columns)}")

    if embed_column not in df.columns:
        raise ValueError(f"Embed column '{embed_column}' not found in Excel!")

    df = df.fillna("")

    kb_texts = []
    kb_meta = []

    for idx, row in df.iterrows():
        desc = clean_text(row[embed_column])

        # Skip weak text
        if not desc or len(desc) < 20:
            continue

        meta = {"row_id": int(idx)}
        for col in df.columns:
            meta[col] = safe_to_str(row[col])

        meta[embed_column] = desc

        kb_texts.append(desc)
        kb_meta.append(meta)

    print(f"KB rows kept after cleaning: {len(kb_texts)}")

    if len(kb_texts) == 0:
        print("ERROR: No text rows remained after cleaning.")
        return None, None

    print(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    print("Creating embeddings...")
    embeddings = []

    for i in range(0, len(kb_texts), batch_size):
        batch = kb_texts[i : i + batch_size]
        batch_emb = embedder.encode(
            batch, show_progress_bar=True, normalize_embeddings=True
        )
        embeddings.append(batch_emb)

    embeddings = np.vstack(embeddings).astype("float32")
    dim = embeddings.shape[1]

    print(f"Embedding shape: {embeddings.shape}")

    # Cosine similarity via Inner Product (since normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(save_dir, "system_kb.faiss")
    meta_path = os.path.join(save_dir, "system_kb_meta.pkl")

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(kb_meta, f)

    print("\n System KB Created Successfully!")
    print(f"Index saved: {index_path}")
    print(f"Meta saved : {meta_path}")

    return index_path, meta_path

# Part 2: Retriever

class SystemKBRetriever:
    def __init__(self, kb_dir="system_kb_store", model_name=DEFAULT_MODEL_NAME):
        index_path = os.path.join(kb_dir, "system_kb.faiss")
        meta_path = os.path.join(kb_dir, "system_kb_meta.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("❌ KB files missing. Build KB first.")

        print(f"\n Loading FAISS index: {index_path}")
        self.index = faiss.read_index(index_path)

        print(f"Loading metadata: {meta_path}")
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

        print(f"Loaded KB rows: {len(self.meta)}")
        print(f"Loading embedder: {model_name}")
        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, query_text: str, top_k: int = 5):
        query_text = str(query_text).strip()
        if not query_text:
            return []

        q_emb = self.embedder.encode([query_text], normalize_embeddings=True).astype(
            "float32"
        )

        scores, idxs = self.index.search(q_emb, top_k)
        results = []

        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue

            results.append({"score": float(score), "meta": self.meta[idx]})

        return results

# Run Full Pipeline (Build + Search)

if __name__ == "__main__":
    EXCEL_PATH = r"C:\Users\mukeshkr\Agentic-AI-Defense-Data-Extraction\data\sample_data.xlsx"
    KB_DIR = "system_kb_store"

    # Step 1: Build KB
    build_system_kb_store_all_columns(
        excel_path=EXCEL_PATH,
        save_dir=KB_DIR,
        model_name=DEFAULT_MODEL_NAME,
        batch_size=64,
        embed_column="Description of Contract",
    )

    # Step 2: Load KB + Retrieve
    r = SystemKBRetriever(kb_dir=KB_DIR)

    query = "Raytheon Co., Tewksbury, Massachusetts, is awarded an $11,172,229 firm-fixed-price modification to previously awarded contract N00024-22-C-5522 for Total Ship Computing Environment Lab hardware for modernization/technical refresh and Conventional Prompt Strike to support DDG 1000-class combat system activation, sustainment and modernization.  Work will be performed in Nashua, New Hampshire (92%); Portsmouth, Rhode Island (4%); and  Tewksbury, Massachusetts (4%), and is expected to be completed by November 2023. Fiscal 2022 other procurement (Navy) funds in the amount of $9,333,203 (84%); and fiscal 2022 research, development, test and evaluation (Navy) funds in the amount of $1,839,026 (16%) will be obligated at time of award and will not expire at the end of the current fiscal year. The Naval Sea Systems Command, Washington, D.C., is the contracting activity."

    hits = r.retrieve(query, top_k=5)

    print("\n" + "=" * 60)
    print("TOP MATCHES")
    print("=" * 60)

    for i, h in enumerate(hits, start=1):
        print(f"\n🔹 Rank: {i}")
        print("Score:", h["score"])
        print("Supplier:", h["meta"].get("Supplier Name", ""))
        print("Market:", h["meta"].get("Market Segment", ""))
        print("System:", h["meta"].get("System Name (Specific)", ""))
        print("Row ID:", h["meta"].get("row_id", ""))
