# Paper Search: Hybrid Search & RAG over CS Research Papers

A search engine and question-answering system built from scratch over ~5,000 arXiv computer science papers. It combines a hand-built BM25 ranking engine with semantic embeddings and a grounded LLM answer layer (RAG).

**Live demo:** [your-render-url-here]

<!-- Add a screenshot here, e.g. ![Demo](docs/demo.png) -->

## What it does

- **Keyword search:** type a query and get ranked results from a custom inverted index and BM25 scorer (no external search library).
- **Ask a question:** get an answer generated only from the retrieved paper abstracts, with citations, and an honest refusal when nothing relevant is found.
- **Typo tolerant:** "machine lerning" is corrected to "machine learning" before retrieval.

## Architecture

```
arXiv abstracts
      |
Tokenizer + Stemmer -> Inverted Index -> BM25 ranking ----+
      |                                                   |
Sentence embeddings (MiniLM) -> cosine similarity --------+-> Hybrid score
                                                          |
                                        Confidence gate (refuse if weak)
                                                          |
                                             LLM (Gemini) with citations
                                                          |
                                    Answer + citation-support check
```

Stack: FastAPI backend, vanilla JS frontend, deployed on Render.

## Key implementation details

- **BM25 from scratch**, including smoothed IDF (never negative) and tuned k1/b parameters.
- **Performance fix:** found and removed an O(N) recomputation in the scoring hot path (average document length was recalculated per candidate per query). Average query time dropped from ~80 ms to ~10 ms (87% faster). See `benchmark.py`.
- **Typo tolerance** via Levenshtein edit distance. Correction runs on raw words before stemming, ties are broken by word frequency, and the corrected query feeds both BM25 and the embedding model.
- **Hybrid retrieval:** a weighted blend of normalized BM25 (exact terms) and embedding cosine similarity (meaning), so related concepts are found even with zero shared words.
- **Confidence-gated generation:** if the best semantic similarity is below a threshold, the system refuses instead of guessing.
- **Citation verification:** checks whether each cited sentence overlaps in wording with its cited source and flags weak support.
- **Graceful degradation:** if the LLM is unavailable or rate-limited, the API returns the most relevant papers with snippets instead of an error. Successful answers are cached in memory.

## Known limitations

- Citation checking is a word-overlap heuristic, not true fact verification.
- Levenshtein counts a transposition (e.g. "nueral") as 2 edits; Damerau-Levenshtein would count 1.
- The dataset is a ~5,000-paper sample of arXiv CS categories, so newer subfields (e.g. graph neural networks) are sparsely covered.
- The Gemini free tier allows only about 20 generation requests per day per model. When that runs out, the app falls back to showing top papers.
- Papers added through `/admin/add_paper` are written to local disk, which is ephemeral on Render's free tier.

## Running locally

Create a `.env` file (never commit it):

```
GEMINI_API_KEY=your_key
ADMIN_KEY=choose_a_secret
GEMINI_MODEL=gemini-3.8-flash
```

Then:

```
pip install -r requirements.txt
python fetch_hf.py          # fetch dataset (skip if data/ is present)
python embeddings.py        # build semantic embeddings (skip if data/ is present)
uvicorn app:app --reload
```

Open http://127.0.0.1:8000.

## Deploying on Render

The repo includes `render.yaml`. In Render choose **New > Blueprint**, select this repo, and enter `GEMINI_API_KEY` and `ADMIN_KEY` when prompted. `data/papers.jsonl` and `data/embeddings.npy` are committed so the service starts without rebuilding them.

## Benchmarks

- 5,003 documents indexed in ~15 s
- Average query time ~10 ms after optimization
- 16,109 unique stemmed tokens in the vocabulary