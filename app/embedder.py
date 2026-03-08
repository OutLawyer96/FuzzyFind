from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self):
        # load once, reuse everywhere — model load takes a couple seconds
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, text: str) -> np.ndarray:
        # single query at inference time
        vec = self.model.encode([text], normalize_embeddings=True,
                                show_progress_bar=False)
        return vec[0]

    def encode_batch(self, texts: list) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64)


embedder = Embedder()
