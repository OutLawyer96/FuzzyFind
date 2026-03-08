import pickle
import pathlib
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


if __name__ == "__main__":
    # load the cleaned docs from prepare_data.py
    docs_path = pathlib.Path("data/cleaned_docs.pkl")
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)

    texts = [d["text"] for d in docs]
    print(f"loaded {len(texts)} docs")

    # MiniLM-L6-v2 is fast enough on CPU and the quality is fine for
    # news text. tried mpnet first but it was too slow for iteration
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # normalizing so we can use dot product instead of full cosine
    # similarity later — small thing but adds up in the cache
    print("encoding embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"sample embedding norm: {np.linalg.norm(embeddings[0]):.4f}  # should be ~1.0")
    print(f"embeddings shape: {embeddings.shape}")

    # save embeddings array
    emb_path = pathlib.Path("data/embeddings.npy")
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)

    # meta is just cleaned_docs without the text key to save memory
    meta = [{k: v for k, v in d.items() if k != "text"} for d in docs]
    meta_path = pathlib.Path("data/doc_metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=4)

    print(f"saved embeddings to {emb_path}")
    print(f"saved metadata to {meta_path}")

    # chromadb setup — cosine space since we normalized the vectors anyway
    client = chromadb.PersistentClient(path="./vectordb")
    collection = client.get_or_create_collection(
        "newsgroups", metadata={"hnsw:space": "cosine"}
    )

    # skip if already populated — re-running this shouldn't re-insert
    if collection.count() > 0:
        print(f"collection already has {collection.count()} docs, skipping insert")
    else:
        batch_size = 500
        ids = [d["doc_id"] for d in docs]
        metadatas = meta

        print("inserting into chromadb...")
        for i in tqdm(range(0, len(docs), batch_size), desc="chroma batches"):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size].tolist()
            batch_docs = texts[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=batch_meta,
            )

        print(f"inserted {collection.count()} docs into chromadb")

    print(f"\ndone.")
    print(f"  docs:       {len(docs)}")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  chroma:     {collection.count()} docs in collection")
