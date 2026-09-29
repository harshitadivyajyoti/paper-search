import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from search_engine import SearchIndex

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading index...")
idx = SearchIndex()
idx.build("data/papers.jsonl")
print("Ready.")


def make_snippet(doc, query_tokens, length=220):
    """Return a short excerpt of the abstract, trying to center it on a matched word."""
    abstract = doc["abstract"]
    lower = abstract.lower()
    pos = -1
    for term in query_tokens:
        found = lower.find(term)
        if found != -1:
            pos = found
            break
    if pos == -1:
        return abstract[:length] + "..."
    start = max(0, pos - length // 3)
    end = min(len(abstract), start + length)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(abstract) else ""
    return prefix + abstract[start:end] + suffix


@app.get("/search")
def search(q: str = Query(..., min_length=1), top_k: int = 10):
    start_time = time.perf_counter()
    query_tokens, _ = idx.candidates(q)
    results = idx.search(q, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    hits = []
    for doc_id, score in results:
        doc = idx.documents[doc_id]
        hits.append({
            "title": doc["title"],
            "snippet": make_snippet(doc, query_tokens),
            "score": round(score, 2),
            "categories": doc.get("categories", ""),
        })

    return {
        "query": q,
        "query_tokens": query_tokens,
        "total_indexed": len(idx.documents),
        "time_ms": round(elapsed_ms, 2),
        "results": hits,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")