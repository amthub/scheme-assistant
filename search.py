import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BM25_WEIGHT = 0.6

df = pd.read_csv("data/chunks_faq_2021.csv")
embeddings = np.load("data/embeddings_faq_2021.npy")

# drop the title chunk (keep only chunks that start with a FAQ number)
keep = [i for i, t in enumerate(df["text"]) if re.match(r"(\d{1,2})\.\s", t)]
df = df.iloc[keep].reset_index(drop=True)
embeddings = embeddings[keep]

model = SentenceTransformer("intfloat/multilingual-e5-small")


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def norm(x):
    x = np.asarray(x, dtype=float)
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)


bm25 = BM25Okapi([tokenize(t) for t in df["text"]])

while True:
    question = input("\nAsk a question (or type q to quit): ").strip()
    if question.lower() == "q":
        break

    q_vec = model.encode("query: " + question, normalize_embeddings=True)
    scores = embeddings @ q_vec
    bm25_scores = bm25.get_scores(tokenize(question))
    hybrid = norm(scores) + BM25_WEIGHT * norm(bm25_scores)
    top3 = np.argsort(hybrid)[::-1][:3]

    for rank, i in enumerate(top3, start=1):
        print(f"\n#{rank}  score {hybrid[i]:.3f}")
        print(df.loc[i, "text"][:300])