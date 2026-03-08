import pickle
import pathlib
import numpy as np
import umap
import skfuzzy as fuzz
import matplotlib
matplotlib.use('Agg')  # no display needed, just saving files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

N_CLUSTERS = 20  # tunable, but 20 matches the actual newsgroup count so it's a reasonable baseline


def find_best_k(data, ks=None):
    if ks is None:
        ks = [10, 15, 20, 25, 30]

    fpcs = []
    fpes = []

    for k in tqdm(ks, desc="testing cluster counts"):
        # m=2.0 is pretty standard, going higher makes everything fuzzy to
        # the point of being useless
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data.T, k, m=2.0, error=1e-5, maxiter=500, init=None
        )
        # FPE: lower = more structured partitioning
        fpe = -np.sum(u * np.log(u + 1e-10)) / u.shape[1]
        fpcs.append(fpc)
        fpes.append(fpe)
        print(f"  k={k:2d}  FPC={fpc:.4f}  FPE={fpe:.4f}")

    # plot both metrics vs k
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ks, fpcs, marker='o')
    ax1.set_title('FPC vs k')
    ax1.set_xlabel('k')
    ax1.set_ylabel('FPC (higher = better)')

    ax2.plot(ks, fpes, marker='o', color='orange')
    ax2.set_title('FPE vs k')
    ax2.set_xlabel('k')
    ax2.set_ylabel('FPE (lower = better)')

    plt.tight_layout()
    out = pathlib.Path("models/cluster_selection.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved cluster selection plot to {out}")

    best_k = ks[int(np.argmax(fpcs))]
    print(f"best k by FPC: {best_k}")
    return best_k


if __name__ == "__main__":
    models_dir = pathlib.Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: UMAP reduction ────────────────────────────────────────────────

    print("loading embeddings...")
    embeddings = np.load("data/embeddings.npy")
    print(f"embeddings shape: {embeddings.shape}")

    # 50d because FCM falls apart in 384d — distances all converge to the
    # same value and centroids stop moving. 50 keeps the structure without the noise
    print("fitting 50d umap reducer...")
    reducer_50d = umap.UMAP(
        n_components=50, metric='cosine',
        n_neighbors=15, min_dist=0.1, random_state=42
    )
    embedded_50d = reducer_50d.fit_transform(embeddings)
    print(f"umap done, reduced shape: {embedded_50d.shape}")

    print("fitting 2d umap reducer for visualization...")
    reducer_2d = umap.UMAP(
        n_components=2, metric='cosine',
        n_neighbors=15, min_dist=0.1, random_state=42
    )
    embedded_2d = reducer_2d.fit_transform(embeddings)
    print(f"2d umap done, shape: {embedded_2d.shape}")

    np.save("data/embeddings_50d.npy", embedded_50d)
    np.save("data/embeddings_2d.npy", embedded_2d)

    with open("models/reducer_50d.pkl", "wb") as f:
        pickle.dump(reducer_50d, f, protocol=4)
    with open("models/reducer_2d.pkl", "wb") as f:
        pickle.dump(reducer_2d, f, protocol=4)

    print("reducers saved")

    # ── STEP 2: cluster count selection ──────────────────────────────────────

    print("\nrunning cluster count selection...")
    find_best_k(embedded_50d)

    # ── STEP 3: final clustering ──────────────────────────────────────────────

    print(f"\nrunning cmeans with k={N_CLUSTERS}, this takes a while...")
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        embedded_50d.T, N_CLUSTERS, m=2.0, error=1e-5, maxiter=1000, init=None
    )
    print(f"cmeans done — FPC: {fpc:.4f}, iterations: {p}")

    np.save("models/cluster_centers.npy", cntr)
    np.save("models/membership_matrix.npy", u)
    print(f"cluster centers: {cntr.shape}, membership matrix: {u.shape}")

    # ── STEP 4: analysis ─────────────────────────────────────────────────────

    # load doc metadata and texts for analysis
    with open("data/cleaned_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    texts = [d["text"] for d in docs]
    label_ids = np.array([d["label"] for d in docs])
    label_names_list = [d["label_name"] for d in docs]

    # u.shape is (N_CLUSTERS, n_docs), transpose for per-doc indexing
    u_T = u.T  # (n_docs, N_CLUSTERS)

    # A) Keywords per cluster via weighted TF-IDF
    print("\ncomputing keywords per cluster...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', min_df=3)
    tfidf_matrix = vectorizer.fit_transform(texts)  # (n_docs, vocab)
    terms = np.array(vectorizer.get_feature_names_out())

    cluster_keywords = {}
    for c in tqdm(range(N_CLUSTERS), desc="keywords"):
        weights = u_T[:, c]  # membership of each doc in this cluster
        # weight each doc's tfidf vector by its membership score
        weighted = tfidf_matrix.T.dot(weights)  # (vocab,)
        top_idx = np.argsort(weighted)[::-1][:15]
        cluster_keywords[c] = terms[top_idx].tolist()

    with open("models/cluster_keywords.pkl", "wb") as f:
        pickle.dump(cluster_keywords, f, protocol=4)
    print("saved cluster_keywords.pkl")

    # B) Cluster-label heatmap
    # this was the most useful sanity check — if clusters are garbage
    # the heatmap looks like noise
    print("building cluster-label heatmap...")
    n_labels = len(set(label_ids))
    heatmap_data = np.zeros((N_CLUSTERS, n_labels))
    for doc_i in range(len(docs)):
        l = label_ids[doc_i]
        heatmap_data[:, l] += u_T[doc_i]  # add membership scores to each cluster row

    # row normalize so dominant label shows clearly regardless of label size
    row_sums = heatmap_data.sum(axis=1, keepdims=True)
    heatmap_data_norm = heatmap_data / (row_sums + 1e-10)

    # load the actual label names from metadata
    with open("data/doc_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    all_label_names = sorted(set(d["label_name"] for d in meta))  # good enough

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        heatmap_data_norm,
        ax=ax,
        cmap='viridis',
        xticklabels=all_label_names,
        yticklabels=[f"c{i}" for i in range(N_CLUSTERS)],
        cbar_kws={"label": "normalized membership"},
    )
    ax.set_title("Cluster vs Newsgroup Label (row normalized membership)")
    ax.set_xlabel("newsgroup")
    ax.set_ylabel("cluster")
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig("models/cluster_label_heatmap.png", dpi=150)
    plt.close()
    print("saved cluster_label_heatmap.png")

    # C) boundary docs — the interesting ones
    # these are docs sitting between clusters, e.g. gun control posts
    # between politics and weapons clusters
    top_memberships = np.sort(u_T, axis=1)[:, ::-1]
    gap = top_memberships[:, 0] - top_memberships[:, 1]

    boundary_mask = gap < 0.15
    boundary_indices = np.where(boundary_mask)[0]
    print(f"\nfound {len(boundary_indices)} boundary docs (gap < 0.15)")

    # sort by smallest gap first — most ambiguous at top
    sorted_boundary = boundary_indices[np.argsort(gap[boundary_indices])]
    top50_boundary = sorted_boundary[:50]

    boundary_docs = []
    for idx in top50_boundary:
        cluster_a = int(np.argsort(u_T[idx])[::-1][0])
        cluster_b = int(np.argsort(u_T[idx])[::-1][1])
        boundary_docs.append({
            "doc_id": docs[idx]["doc_id"],
            "label_name": docs[idx]["label_name"],
            "gap": float(gap[idx]),
            "top_cluster": cluster_a,
            "second_cluster": cluster_b,
            "top_keywords_a": cluster_keywords[cluster_a][:5],
            "top_keywords_b": cluster_keywords[cluster_b][:5],
            "text_snippet": docs[idx]["text"][:200],
        })

    with open("models/boundary_docs.pkl", "wb") as f:
        pickle.dump(boundary_docs, f, protocol=4)
    print("saved boundary_docs.pkl")

    print("\ntop 10 boundary docs:")
    for bd in boundary_docs[:10]:
        print(f"  [{bd['label_name']}] gap={bd['gap']:.4f}  "
              f"c{bd['top_cluster']}({bd['top_keywords_a'][:3]}) vs "
              f"c{bd['second_cluster']}({bd['top_keywords_b'][:3]})")

    # D) 2D scatter plot colored by dominant cluster
    print("\nplotting 2d umap scatter...")
    dominant_cluster = np.argmax(u_T, axis=1)
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(
        embedded_2d[:, 0], embedded_2d[:, 1],
        c=dominant_cluster, cmap='tab20', s=1.5, alpha=0.6
    )
    plt.colorbar(scatter, ax=ax, label="cluster")
    ax.set_title("UMAP 2D — colored by dominant cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig("models/umap_cluster_plot.png", dpi=150)
    plt.close()
    print("saved umap_cluster_plot.png")

    # E) Uncertainty histogram
    # if this spikes near 1.0, clusters are clean. if it's flat,
    # the corpus is genuinely ambiguous
    max_memberships = u_T.max(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_memberships, bins=50, edgecolor='black', linewidth=0.4)
    ax.set_title("Distribution of max membership per doc")
    ax.set_xlabel("max membership score")
    ax.set_ylabel("doc count")
    ax.axvline(0.5, color='red', linestyle='--', linewidth=0.8, label='0.5')
    ax.legend()
    plt.tight_layout()
    plt.savefig("models/uncertainty_histogram.png", dpi=150)
    plt.close()
    print("saved uncertainty_histogram.png")

    # ── STEP 5: summary table ─────────────────────────────────────────────────

    print("\n" + "─" * 72)
    print(f"{'Cluster':<9} {'Top 3 keywords':<36} {'Docs':>6} {'Avg max membership':>18}")
    print("─" * 72)
    for c in range(N_CLUSTERS):
        top3 = ", ".join(cluster_keywords[c][:3])
        # doc count = docs where this is the dominant cluster
        doc_count = int((dominant_cluster == c).sum())
        avg_max = float(u_T[dominant_cluster == c, c].mean()) if doc_count > 0 else 0.0
        print(f"  c{c:<6d} {top3:<36} {doc_count:>6} {avg_max:>18.4f}")
    print("─" * 72)
