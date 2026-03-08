import chromadb
import numpy as np
from pathlib import Path


class VectorStore:
    def __init__(self):
        if not Path("./vectordb").exists():
            raise RuntimeError("vectordb not found — run generate_embeddings.py first")

        self.client = chromadb.PersistentClient(path="./vectordb")
        self.collection = self.client.get_collection("newsgroups")

    def search(self, embedding: np.ndarray, n: int = 5) -> dict:
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for i in range(len(results['ids'][0])):
            hits.append({
                "doc_id": results['ids'][0][i],
                "snippet": results['documents'][0][i][:300],
                "label": results['metadatas'][0][i]['label_name'],
                # chromadb cosine distance = 1 - similarity
                "similarity": round(1.0 - results['distances'][0][i], 4)
            })

        return {"hits": hits}


vector_store = VectorStore()
