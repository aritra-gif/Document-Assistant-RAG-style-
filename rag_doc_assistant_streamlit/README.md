# Document Assistant (RAG-style) — Streamlit + Notebooks

This repo is a simple, beginner-friendly “RAG-style” assistant:
- load documents
- chunk them
- build an index (TF‑IDF)
- ask questions and get answers with citations (chunk IDs)

No paid LLM required for the demo.

---

## What’s inside
- `app.py` → Streamlit UI (upload docs, build index, ask questions)
- `notebooks/01_ingest_and_index.ipynb` → chunking + indexing explained
- `notebooks/02_retrieval_and_eval.ipynb` → retrieval + MMR + tiny evaluation
- `data/docs/` → sample docs

---

## Setup
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```

---

## Interview notes (plain language)
- RAG = retrieve evidence first, then answer using that evidence
- chunking matters a lot
- measure retrieval (Recall@K / MRR)
- retrieved text is untrusted (basic injection patterns should be filtered)
