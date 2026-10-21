# PM-Kisan Scheme Assistant

A question-answering assistant for India's PM-Kisan scheme. You ask a question in English, Hindi or Hinglish, and it answers in the same language, using only the official 2021 PM-Kisan FAQ.

## How it works

1. **Search.** The question is matched against 27 FAQ chunks with two methods and the scores are blended (hybrid search):
   - Semantic search with `multilingual-e5-small` embeddings
   - Keyword search with BM25
2. **Answer.** The top 3 chunks go to Gemini (`gemini-3.8-flash`, free tier), which writes a short answer from that text only. If the FAQ does not contain the answer, it says so.
3. **UI.** A small Streamlit page lets you type a question in the browser and shows the source FAQ entries.

## Results

I measured whether the right FAQ appears in the top 3 results (hit@3) on my own test questions (`data/test_questions.csv`).

| Test set | Method | hit@3 |
|---|---|---|
| 12 questions | e5 only | 0.92 |
| 12 questions | Hybrid (e5 + BM25) | 1.00 |
| 27 questions | e5 only (first run) | 0.85 |
| 27 questions | Hybrid (e5 + BM25) | 0.93 |

Hybrid search also lifted the Hinglish hit@1 from 0.25 to 0.50 on the 12-question set.

### What did not work
A cross-encoder reranker (`BAAI/bge-reranker-base`) made results worse (hit@3 fell from 0.92 to 0.75), because it does not understand Romanized Hindi. It is still in `evaluate.py` but switched off (`USE_RERANKER = False`).

### Known weaknesses
- Two Hinglish test questions still miss the right FAQ in the top 3. Everyday wording like "galat jaankari" does not match formal FAQ wording like "incorrect declaration".
- The test set is small (27 questions, 9 per language), so scores move a lot when one question changes.
- The assistant only knows the 2021 FAQ, so newer rule changes are not covered.

## Setup

1. Create a virtual environment and install the packages:
```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
```
2. Get a free Gemini API key from Google AI Studio and put it in a file named `.env`:
```
   GEMINI_API_KEY=your-key-here
```
   The `.env` file is ignored by git, so the key is never uploaded.

## Run

- Web page: `streamlit run app.py --server.fileWatcherType none`
- Command line: `python answer.py`
- Evaluation: `python evaluate.py`

The file watcher is switched off because Streamlit's watcher fails on the `transformers` library when `torchvision` is not installed.

## Project files

- `fetch.py`, `extract_text.py`, `make_chunks.py`, `make_embeddings.py`: build the data (download, extract text, chunk, embed)
- `search.py`: hybrid search in the terminal
- `answer.py`: hybrid search plus Gemini answers in the terminal
- `app.py`: Streamlit web page
- `evaluate.py`: measures hit@1 and hit@3 for e5, BM25 and hybrid
- `data/`: chunks, embeddings, test questions and source log