import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

chunks = pd.read_csv("data/chunks_faq_2021.csv")
embeddings = np.load("data/embeddings_faq_2021.npy")
tests = pd.read_csv("data/test_questions.csv", encoding="utf-8")
keep = [i for i, t in enumerate(chunks["text"]) if re.match(r"(\d{1,2})\.\s", t)]
chunks = chunks.iloc[keep].reset_index(drop=True)
embeddings = embeddings[keep]
model = SentenceTransformer("intfloat/multilingual-e5-small")

from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
USE_RERANKER = False

BM25_WEIGHT = 0.6

def norm(x):
    x = np.asarray(x, dtype=float)
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)

from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"\w+", text.lower())

bm25 = BM25Okapi([tokenize(t) for t in chunks["text"]])

chunk_faq_no = []
for text in chunks["text"]:
    match = re.match(r"(\d{1,2})\.\s", text)
    chunk_faq_no.append(int(match.group(1)) if match else 0)

results = []
for i, row in tests.iterrows():
    correct = {int(n) for n in str(row["correct_faq_no"]).split("|")}
    q_vec = model.encode("query: " + row["question"], normalize_embeddings=True)
    scores = embeddings @ q_vec
    bm25_scores = bm25.get_scores(tokenize(row["question"]))
    bm25_top3 = np.argsort(bm25_scores)[::-1][:3]
    bm25_top3_faq = [chunk_faq_no[j] for j in bm25_top3]
    hybrid = norm(scores) + BM25_WEIGHT * norm(bm25_scores)
    hybrid_top3 = np.argsort(hybrid)[::-1][:3]
    hybrid_top3_faq = [chunk_faq_no[j] for j in hybrid_top3]
    top5 = np.argsort(scores)[::-1][:5]
    pairs = [(row["question"], chunks["text"][j]) for j in top5]
    rerank_scores = reranker.predict(pairs) if USE_RERANKER else scores[top5]
    order = np.argsort(rerank_scores)[::-1][:3]
    top3 = [top5[k] for k in order]
    top3_faq = [chunk_faq_no[j] for j in top3]
    results.append({
        "test": i + 1,
        "language": row["language"],
        "question": row["question"],
        "correct": sorted(correct),
        "top3_faq_no": top3_faq,
        "top3_scores": [round(float(rerank_scores[k]), 3) for k in order],
        "hit_at_1": top3_faq[0] in correct,
        "hit_at_3": any(n in correct for n in top3_faq),
        "bm25_hit_at_1": bm25_top3_faq[0] in correct,
        "bm25_hit_at_3": any(n in correct for n in bm25_top3_faq),
        "hybrid_hit_at_1": hybrid_top3_faq[0] in correct,
        "hybrid_hit_at_3": any(n in correct for n in hybrid_top3_faq),
         "hybrid_top3_faq": hybrid_top3_faq,
    })

res = pd.DataFrame(results)

print("Chunks with no FAQ number parsed:", chunk_faq_no.count(0))
print()
for _, r in res[~res["hit_at_1"]].iterrows():
    print(r["language"], "| test", r["test"], "|", r["question"])
    print("  expected:", r["correct"], "| got (faq_no, score):", list(zip(r["top3_faq_no"], r["top3_scores"])))
print()
print(res.groupby("language")[["hit_at_1", "hit_at_3"]].mean().round(2))
print()
print("Overall hit@3:", round(res["hit_at_3"].mean(), 2))
print()
print("BM25 only:")
print(res.groupby("language")[["bm25_hit_at_1", "bm25_hit_at_3"]].mean().round(2))
print()
print("Hybrid (e5 + BM25):")
print(res.groupby("language")[["hybrid_hit_at_1", "hybrid_hit_at_3"]].mean().round(2))
print("Hybrid overall hit@3:", round(res["hybrid_hit_at_3"].mean(), 2))
print()
print("Hybrid misses (right FAQ not in top 3):")
for _, r in res[~res["hybrid_hit_at_3"]].iterrows():
    print(r["language"], "| test", r["test"], "|", r["question"])
    print("  expected:", r["correct"], "| hybrid top 3:", r["hybrid_top3_faq"])
res.to_csv("data/eval_baseline.csv", index=False)
