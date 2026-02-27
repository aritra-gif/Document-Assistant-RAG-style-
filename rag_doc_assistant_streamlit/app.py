import streamlit as st
import re
from pathlib import Path
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Document Assistant", layout="wide")
st.title("📄 Document Assistant (RAG-style demo)")

ART = Path("artifacts/index.joblib")
Path("artifacts").mkdir(exist_ok=True)

st.sidebar.header("Documents")
use_samples = st.sidebar.checkbox("Use sample docs from data/docs", value=True)
uploaded = st.sidebar.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)

def chunk_text(text: str, max_chars=700, overlap=80):
    parts = re.split(r"\n\s*\n", text.strip())
    chunks, buf = [], ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buf) + len(part) + 2 <= max_chars:
            buf = (buf + "\n\n" + part).strip()
        else:
            if buf:
                chunks.append(buf)
            while len(part) > max_chars:
                chunks.append(part[:max_chars])
                part = part[max_chars - overlap:]
            buf = part
    if buf:
        chunks.append(buf)
    return chunks

def build_index(docs):
    chunks = []
    for doc_id, text in docs:
        for i, ch in enumerate(chunk_text(text)):
            chunks.append({"doc_id": doc_id, "chunk_id": f"{doc_id}::chunk{i}", "text": ch})
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vec.fit_transform([c["text"] for c in chunks])
    dump({"vectorizer": vec, "matrix": X, "chunks": chunks}, ART)
    return len(chunks)

INJECTION_PATTERNS = [r"ignore\s+previous", r"system\s+prompt", r"developer\s+message", r"do\s+not\s+follow"]
def looks_like_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)

def mmr_select(query_vec, doc_vecs, lambda_=0.6, k=6):
    def cos(a,b):
        a=a/(np.linalg.norm(a)+1e-9); b=b/(np.linalg.norm(b)+1e-9)
        return float(a@b)
    selected, candidates = [], list(range(doc_vecs.shape[0]))
    while candidates and len(selected) < k:
        best, best_score = None, -1e18
        for i in candidates:
            rel = cos(query_vec, doc_vecs[i])
            div = 0.0 if not selected else max(cos(doc_vecs[i], doc_vecs[j]) for j in selected)
            score = lambda_*rel - (1-lambda_)*div
            if score > best_score:
                best_score, best = score, i
        selected.append(best); candidates.remove(best)
    return selected

def retrieve(artifact, question, k_candidates=20, k_final=6):
    vec, X, chunks = artifact["vectorizer"], artifact["matrix"], artifact["chunks"]
    q = vec.transform([question])
    sims = cosine_similarity(q, X).reshape(-1)
    cand_idx = sims.argsort()[::-1][:min(k_candidates, len(sims))]
    doc_vecs = X[cand_idx].toarray()
    qv = q.toarray().reshape(-1)
    picked = mmr_select(qv, doc_vecs, lambda_=0.6, k=min(k_final, len(cand_idx)))
    picked_idx = [cand_idx[i] for i in picked]
    out = []
    for i in picked_idx:
        ch = chunks[i]
        if looks_like_injection(ch["text"]):
            continue
        out.append({**ch, "score": float(sims[i])})
    return out

docs = []
if use_samples:
    for p in Path("data/docs").glob("*.txt"):
        docs.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
for uf in uploaded or []:
    docs.append((uf.name, uf.read().decode("utf-8", errors="ignore")))

st.sidebar.header("Index")
if st.sidebar.button("Build index", type="primary", disabled=(len(docs)==0)):
    n = build_index(docs)
    st.sidebar.success(f"Index built ({n} chunks).")

artifact = load(ART) if ART.exists() else None
if artifact is None:
    st.info("Add docs and click **Build index** in the sidebar.")
else:
    st.sidebar.info(f"Index loaded ({len(artifact['chunks'])} chunks).")
    st.subheader("Ask a question")
    q = st.text_input("Question", value="What is the refund policy?")
    if st.button("Search + Answer", type="primary"):
        hits = retrieve(artifact, q)
        if not hits:
            st.error("No safe evidence found for this question.")
        else:
            top = hits[0]
            snippet = top["text"].replace("\n"," ")
            snippet = (snippet[:260] + "...") if len(snippet) > 260 else snippet
            st.success("Answer (grounded from docs):")
            st.write(snippet)
            st.caption("Citations: " + ", ".join([h["chunk_id"] for h in hits[:3]]))

            with st.expander("Show retrieved chunks"):
                for h in hits:
                    st.markdown(f"**{h['chunk_id']}** (score={h['score']:.3f})")
                    st.write(h["text"])
                    st.divider()
