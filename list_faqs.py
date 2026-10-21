import re

import pandas as pd

df = pd.read_csv("data/chunks_faq_2021.csv")
for t in df["text"]:
    if re.match(r"(\d{1,2})\.\s", t):
        print(t[:110].replace("\n", " "))