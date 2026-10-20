import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/chunks_faq_2021.csv")
embeddings = np.load("data/embeddings_faq_2021.npy")
model = SentenceTransformer("intfloat/multilingual-e5-small")

while True:
    question = input("\nAsk a question (or type q to quit): ").strip()
    if question.lower() == "q":
        break

    q_vec = model.encode("query: " + question, normalize_embeddings=True)
    scores = embeddings @ q_vec
    top3 = np.argsort(scores)[::-1][:3]

    for rank, i in enumerate(top3, start=1):
        print(f"\n#{rank}  score {scores[i]:.3f}")
        print(df.loc[i, "text"][:300])