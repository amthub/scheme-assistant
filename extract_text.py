from pathlib import Path
from pypdf import PdfReader

OUT_DIR = Path("data/text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in sorted(Path("data/raw").glob("*.pdf")):
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    out_path = OUT_DIR / f"{pdf_path.stem}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} ({len(text)} characters)")