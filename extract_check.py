from pathlib import Path
from pypdf import PdfReader

for pdf_path in sorted(Path("data/raw").glob("*.pdf")):
    reader = PdfReader(pdf_path)
    pages = max(len(reader.pages), 1)
    chars = sum(len(page.extract_text() or "") for page in reader.pages)
    print(f"{pdf_path.name}: {pages} pages, {chars} characters, {chars // pages} per page")