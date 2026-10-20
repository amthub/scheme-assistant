import re
from pathlib import Path

import pandas as pd

text = Path("data/text/pmkisan_faq_2021.txt").read_text(encoding="utf-8")

parts = re.split(r"\n(?=\d{1,2}\.\s)", text)

rows = []
for part in parts:
    clean = " ".join(part.split())
    if len(clean) < 40:
        continue
    rows.append({
        "doc": "pmkisan_faq_2021",
        "chunk_id": len(rows) + 1,
        "text": clean,
    })

df = pd.DataFrame(rows)
df.to_csv("data/chunks_faq_2021.csv", index=False, encoding="utf-8")

print(f"Made {len(df)} chunks")
print(df["text"].str.len().describe())
print()
print(df.loc[1, "text"][:300])