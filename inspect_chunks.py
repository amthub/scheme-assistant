import re
import pandas as pd

chunks = pd.read_csv("data/chunks_faq_2021.csv")
for idx, text in enumerate(chunks["text"]):
    m = re.match(r"(\d{1,2})\.\s", text)
    n = int(m.group(1)) if m else 0
    if n in (0, 12, 13):
        print("row", idx, "| faq", n, "| chars", len(text))
        print(text[:400])
        print("-" * 40)