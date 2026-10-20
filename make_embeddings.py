import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/chunks_faq_2021.csv")

model = SentenceTransformer("intfloat/multilingual-e5-small")

texts = ["passage: " + t for t in df["text"]]
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

np.save("data/embeddings_faq_2021.npy", embeddings)
print(embeddings.shape)