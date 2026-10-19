import truststore
truststore.inject_into_ssl()

import requests
from bs4 import BeautifulSoup

URL = "https://pmkisan.gov.in/"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

response = requests.get(URL, headers=headers, timeout=30)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()

text = soup.get_text(separator="\n", strip=True)

with open("pmkisan_raw.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Downloaded {len(response.text)} characters of HTML")
print(f"Extracted {len(text)} characters of text")
print(text[:500])