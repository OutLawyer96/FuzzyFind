import re
import pickle
import pathlib
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm


def clean_text(text):
    text = text.lower()
    # strip emails — they show up constantly and wreck embeddings
    text = re.sub(r'\S+@\S+', '', text)
    # strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # remove lines that are just dashes, equals signs, or bare numbers
    # these are signature separators, super common in usenet posts
    lines = text.splitlines()
    lines = [l for l in lines if not re.fullmatch(r'[\-=\d\s]+', l.strip()) or not l.strip()]
    text = ' '.join(lines)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if __name__ == "__main__":
    # removing headers/footers because they have email addresses and
    # metadata that would bleed into embeddings
    raw = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    texts = raw.data
    labels = raw.target
    label_names = raw.target_names

    n_original = len(texts)

    # clean everything, tqdm so we know it's actually doing something
    cleaned = []
    for text, label in tqdm(zip(texts, labels), total=n_original, desc="cleaning"):
        cleaned.append((clean_text(text), label))

    # drop exact duplicates — there are reposts in this dataset, found ~200 dupes on first run
    seen = set()
    deduped = []
    for text, label in cleaned:
        if text not in seen:
            seen.add(text)
            deduped.append((text, label))

    n_dedup = len(deduped)

    # tried 40 words first, still got garbage docs, 60 seems right
    filtered = [(text, label) for text, label in deduped if len(text.split()) >= 60]

    n_final = len(filtered)
    avg_words = sum(len(text.split()) for text, _ in filtered) / n_final  # good enough

    docs = [
        {
            "text": text,
            "label": int(label),
            "label_name": label_names[label],
            "doc_id": f"doc_{i+1:05d}",
        }
        for i, (text, label) in enumerate(filtered)
    ]

    out_path = pathlib.Path("data/cleaned_docs.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(docs, f, protocol=4)

    print(f"started with {n_original} docs")
    print(f"after dedup: {n_dedup}")
    print(f"after length filter: {n_final}")
    print(f"avg words: {avg_words:.0f}")
