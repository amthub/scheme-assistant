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

chunk_faq_no = []
for text in chunks["text"]:
    match = re.match(r"(\d{1,2})\.\s", text)
    chunk_faq_no.append(int(match.group(1)) if match else 0)

results = []
for i, row in tests.iterrows():
    correct = {int(n) for n in str(row["correct_faq_no"]).split("|")}
    q_vec = model.encode("query: " + row["question"], normalize_embeddings=True)
    scores = embeddings @ q_vec
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

res.to_csv("data/eval_baseline.csv", index=False)
