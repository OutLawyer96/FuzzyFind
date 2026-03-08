import numpy as np
import time
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    result: dict
    cluster: int
    membership: np.ndarray
    ts: float
    hits: int = 0


class SemanticCache:
    """
    Semantic cache built from scratch.

    Entries are bucketed by dominant cluster so lookup only scans
    docs in the same cluster as the query. With k=20 clusters and
    N cached entries that's N/20 comparisons on average instead of N.

    Similarity is just dot product — works because embeddings are
    already L2 normalized, so dot product == cosine similarity.

    Threshold behavior (this is the main thing to tune):
      0.70 — too loose, unrelated queries can match
      0.80 — paraphrases match, different questions don't
      0.82 — default, sweet spot found by testing
      0.90 — only near-identical phrasing matches
      0.95 — barely anything hits, cache is almost useless
    """

    def __init__(self, threshold: float = 0.82):
        self.threshold = threshold
        self.buckets: dict[int, list[CacheEntry]] = {}
        self.hits = 0
        self.misses = 0
        self._lookup_time = 0.0
        self._n_lookups = 0

    def lookup(self, embedding: np.ndarray, cluster: int):
        t0 = time.perf_counter()
        self._n_lookups += 1

        best, best_score = None, 0.0
        for entry in self.buckets.get(cluster, []):
            score = float(np.dot(embedding, entry.embedding))
            if score > best_score:
                best_score = score
                best = entry

        self._lookup_time += time.perf_counter() - t0

        if best_score >= self.threshold:
            self.hits += 1
            best.hits += 1
            return best, best_score

        self.misses += 1
        return None, 0.0

    def store(self, query, embedding, result, cluster, membership):
        entry = CacheEntry(
            query=query,
            embedding=embedding.copy(),
            result=result,
            cluster=cluster,
            membership=membership.copy(),
            ts=time.time()
        )
        self.buckets.setdefault(cluster, []).append(entry)

    def flush(self):
        self.buckets.clear()
        self.hits = 0
        self.misses = 0
        self._lookup_time = 0.0
        self._n_lookups = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        n_entries = sum(len(b) for b in self.buckets.values())
        return {
            "total_entries": n_entries,
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "avg_lookup_ms": round(self._lookup_time / self._n_lookups * 1000, 3)
                             if self._n_lookups else 0.0,
            "per_cluster": {str(k): len(v) for k, v in sorted(self.buckets.items())}
        }
