import os
import time
import re
from dotenv import load_dotenv
from google import genai
from search_engine import SearchIndex, SemanticSearch

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CONFIDENCE_THRESHOLD = 0.35   # cosine similarity below this = "probably not relevant"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def check_citation_support(answer_text, sources, overlap_threshold=0.15):
    flags = []
    sentences = re.split(r'(?<=[.!?])\s+', answer_text)
    for sent in sentences:
        cites = re.findall(r'\[(\d+)\]', sent)
        if not cites:
            continue
        sent_words = set(re.findall(r'\w+', sent.lower())) - {"the", "a", "an", "is", "of", "in", "to"}
        for c in cites:
            idx_num = int(c) - 1
            if idx_num < 0 or idx_num >= len(sources):
                continue
            abstract_words = set(re.findall(r'\w+', sources[idx_num]["abstract"].lower()))
            if not sent_words:
                continue
            overlap = len(sent_words & abstract_words) / len(sent_words)
            if overlap < overlap_threshold:
                flags.append({"sentence": sent.strip(), "citation": f"[{c}]", "overlap": round(overlap, 2)})
    return flags


def answer_question(query, idx, semantic, top_k=5):
    results = idx.hybrid_search(query, semantic, top_k=top_k)
    best_semantic_score = results[0][2] if results else 0.0

    sources = []
    for doc_id, score, sem in results:
        doc = idx.documents[doc_id]
        sources.append({"title": doc["title"], "abstract": doc["abstract"]})

    if best_semantic_score < CONFIDENCE_THRESHOLD:
        return {
            "answer": "I couldn't find papers in this collection that clearly relate to your question. "
                      "The closest matches had low relevance, so I'm not going to guess. "
                      "Try rephrasing, or this topic may not be covered in the current dataset.",
            "sources": [],
            "confidence": round(float(best_semantic_score), 3),
            "low_confidence": True,
            "citation_flags": [],
        }

    context = "\n\n".join(f"[{i+1}] {s['title']}\n{s['abstract']}" for i, s in enumerate(sources))
    prompt = f"""Answer the question using ONLY the paper excerpts below. Cite papers as [1], [2], etc. If the excerpts don't contain a good answer, say so honestly.

Excerpts:
{context}

Question: {query}"""

    answer = None
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            answer = response.text
            if answer:
                break
        except Exception as e:
            last_error = e
            print(f"  Gemini error (attempt {attempt+1}): {type(e).__name__}: {e}")
            msg = str(e)
            transient = any(code in msg for code in ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if not transient:
                break                      # wrong model / bad key: retrying is pointless
            time.sleep(2 * (attempt + 1))
    if answer is None:
        print("LLM failed:", last_error)
        fallback = "\n\n".join(
            f"[{i+1}] {s['title']}: {s['abstract'][:300]}..." for i, s in enumerate(sources[:3])
        )
        return {
            "answer": "The AI model couldn't generate a summary right now. Here are the most relevant papers:\n\n" + fallback,
            "sources": [s["title"] for s in sources],
            "confidence": round(float(best_semantic_score), 3),
            "low_confidence": False,
            "citation_flags": [],
        }
    flags = check_citation_support(answer, sources)

    return {
        "answer": answer,
        "sources": [s["title"] for s in sources],
        "confidence": round(float(best_semantic_score), 3),
        "low_confidence": False,
        "citation_flags": flags,
    }


if __name__ == "__main__":
    idx = SearchIndex()
    idx.build("data/papers.jsonl")
    semantic = SemanticSearch()

    for query in ["neural network approaches to time series forecasting", "How do black holes form?"]:
        print(f"\n=== {query} ===")
        result = answer_question(query, idx, semantic)
        print("Confidence:", result["confidence"], "| Low confidence:", result["low_confidence"])
        print("ANSWER:\n", result["answer"])
        if result["citation_flags"]:
            print("\n⚠ Citation check flagged:")
            for fl in result["citation_flags"]:
                print(f"  {fl['citation']} (overlap {fl['overlap']}): \"{fl['sentence'][:80]}...\"")