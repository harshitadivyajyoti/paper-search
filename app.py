import os
import time
from fastapi import FastAPI, Query, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from search_engine import SearchIndex, SemanticSearch
from rag import answer_question

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading search index...")
idx = SearchIndex()
idx.build("data/papers.jsonl")
print("Loading semantic model...")
semantic = SemanticSearch()
print("Ready.")

ADMIN_KEY = os.getenv("ADMIN_KEY")


def make_snippet(doc, query_tokens, length=220):
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


class NewPaper(BaseModel):
    title: str
    abstract: str
    categories: str = ""


class AskRequest(BaseModel):
    question: str


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


@app.post("/ask")
def ask(req: AskRequest):
    return answer_question(req.question, idx, semantic)


@app.post("/admin/add_paper")
def add_paper(paper: NewPaper, x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    doc = {
        "id": f"manual-{len(idx.documents)}",
        "title": paper.title,
        "abstract": paper.abstract,
        "categories": paper.categories,
        "authors": "",
    }
    doc_id = idx.add_paper_live(doc)
    semantic.add_document(paper.title + ". " + paper.abstract)
    return {"status": "added", "doc_id": doc_id, "total_papers": len(idx.documents)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")