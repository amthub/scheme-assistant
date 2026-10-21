import os
import re
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL = "gemini-3.8-flash"
BM25_WEIGHT = 0.6

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

df = pd.read_csv("data/chunks_faq_2021.csv")
embeddings = np.load("data/embeddings_faq_2021.npy")

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


def ask_gemini(prompt):
    for attempt in range(3):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt).text
        except Exception as e:
            print("Gemini error:", str(e)[:150])
            time.sleep(5)
    return None


while True:
    question = input("\nAsk a question (or type q to quit): ").strip()
    if question.lower() == "q":
        break

    q_vec = model.encode("query: " + question, normalize_embeddings=True)
    scores = embeddings @ q_vec
    bm25_scores = bm25.get_scores(tokenize(question))
    hybrid = norm(scores) + BM25_WEIGHT * norm(bm25_scores)
    top3 = np.argsort(hybrid)[::-1][:3]

    context = "\n\n".join(df.loc[i, "text"] for i in top3)
    prompt = (
        "You are an assistant for the PM-Kisan scheme. Answer the question "
        "using ONLY the FAQ text below. If the FAQ text does not contain the "
        "answer, say you do not have that information. Reply in the same "
        "language and script as the question (Hinglish in English letters, "
        "Hindi in Devanagari, English in English). Keep it short.\n\n"
        f"FAQ text:\n{context}\n\nQuestion: {question}"
    )

    answer = ask_gemini(prompt)
    print("\n" + (answer or "Could not get an answer, try again in a minute."))
    print("\nSources: FAQ", ", ".join(df.loc[i, "text"].split(".")[0] for i in top3))