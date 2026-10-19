import truststore
truststore.inject_into_ssl()

from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
RAW_DIR = Path("data/raw")
LOG_FILE = Path("data/sources_log.csv")


def fetch(url: str, name: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    extension = ".pdf" if "pdf" in content_type else ".html"
    path = RAW_DIR / f"{name}{extension}"
    path.write_bytes(response.content)

    row = pd.DataFrame([{
        "name": name,
        "url": url,
        "saved_as": str(path),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "size_kb": round(len(response.content) / 1024, 1),
    }])
    row.to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(),
               index=False, encoding="utf-8")
    print(f"Saved {path} ({row.loc[0, 'size_kb']} KB)")


if __name__ == "__main__":
    fetch("https://pmkisan.gov.in/", "pmkisan_home")