import numpy as np
import pickle
from pathlib import Path


class ClusterPredictor:
    def __init__(self):
        models = Path("./models")

        # umap_reducer_50d.pkl matches the name written by run_clustering.py
        needed = ['reducer_50d.pkl', 'cluster_centers.npy', 'cluster_keywords.pkl']
        for f in needed:
            if not (models / f).exists():
                raise RuntimeError(f"{f} not found — run run_clustering.py first")

        self.reducer = pickle.load(open(models / 'reducer_50d.pkl', 'rb'))
        self.centers = np.load(models / 'cluster_centers.npy')  # (k, 50)
        self.keywords = pickle.load(open(models / 'cluster_keywords.pkl', 'rb'))
        self.m = 2.0  # has to match what was used in training

    def predict(self, embedding: np.ndarray):
        # project into the same 50d space the centroids live in
        reduced = self.reducer.transform(embedding.reshape(1, -1))[0]

        # compute distances to all centroids
        dists = np.linalg.norm(self.centers - reduced, axis=1)
        dists = np.maximum(dists, 1e-10)  # this kept causing div/0 errors

        # FCM membership formula: u_i = 1 / sum_k((d_i/d_k)^(2/(m-1)))
        exp = 2.0 / (self.m - 1.0)
        ratios = (dists[:, None] / dists[None, :]) ** exp
        membership = 1.0 / ratios.sum(axis=1)
        membership /= membership.sum()  # renormalize for float stability

        dominant = int(np.argmax(membership))
        return membership, dominant

    def keywords_for(self, cluster_id: int) -> list:
        return self.keywords.get(cluster_id, [])


cluster_predictor = ClusterPredictor()
