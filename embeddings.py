import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good quality for its size
EMB_FILE = "data/embeddings.npy"
IDS_FILE = "data/embeddings_ids.json"


def build_embeddings(jsonl_path="data/papers.jsonl"):
    model = SentenceTransformer(MODEL_NAME)

    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    texts = [d["title"] + ". " + d["abstract"] for d in docs]
    print(f"Encoding {len(texts)} documents...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=64)

    np.save(EMB_FILE, vectors)
    with open(IDS_FILE, "w") as f:
        json.dump(list(range(len(docs))), f)
    print(f"Saved embeddings: {vectors.shape}")


if __name__ == "__main__":
    build_embeddings()