import json
import re
import math
from collections import defaultdict
from nltk.stem import PorterStemmer
from collections import Counter
import numpy as np

stemmer = PorterStemmer()
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "if", "than", "that", "this", "these", "those",
    "it", "its", "we", "our", "their", "which", "can", "using", "based",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return [stemmer.stem(t) for t in tokens]


class SearchIndex:
    def __init__(self):
        self.documents = []                   # list of {id, title, abstract}
        self.doc_tokens = []                  # tokenized tokens per doc, same order as documents
        self.inverted_index = defaultdict(set)  # word -> set of doc positions
        self.doc_freq = defaultdict(int)      # word -> number of docs containing it
        self.doc_len = []                     # token count per doc (needed for BM25 tomorrow)
        self.term_freqs = []
        self.avg_doc_len = 0.0
        self.raw_vocab = Counter()
        self._corr_cache = {}

    def add_document(self, doc):
        text = doc["title"] + " " + doc["abstract"]
        tokens = tokenize(text)
        doc_id = len(self.documents)

        self.documents.append(doc)
        self.doc_tokens.append(tokens)
        self.doc_len.append(len(tokens))
        self.term_freqs.append(Counter(tokens))
        self.raw_vocab.update(TOKEN_RE.findall(text.lower()))

        seen_in_this_doc = set()
        for tok in tokens:
            self.inverted_index[tok].add(doc_id)
            if tok not in seen_in_this_doc:
                self.doc_freq[tok] += 1
                seen_in_this_doc.add(tok)

    def build(self, jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                self.add_document(doc)
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len)
        print(f"Indexed {len(self.documents)} documents, "
              f"{len(self.inverted_index)} unique words")

    def candidates(self, query):
        raw_tokens = tokenize(query)
        query_tokens = [self.correct_term(t) for t in raw_tokens]
        result_ids = set()
        for tok in query_tokens:
            result_ids |= self.inverted_index.get(tok, set())
        return query_tokens, result_ids

    def idf(self, term):
        n_docs = len(self.documents)
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        # standard BM25 IDF formula (never goes negative, unlike raw log(N/df))
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    def bm25_score(self, doc_id, query_tokens, k1=1.5, b=0.75):
        avg_len = self.avg_doc_len
        doc_len = self.doc_len[doc_id]
        tokens_in_doc = self.doc_tokens[doc_id]

        score = 0.0
        for term in query_tokens:
            freq = self.term_freqs[doc_id].get(term, 0)     # how many times term appears in this doc
            if freq == 0:
                continue
            idf = self.idf(term)
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * (numerator / denominator)
        return score

    def search(self, query, top_k=10):
        query_tokens, candidate_ids = self.candidates(query)
        scored = [(doc_id, self.bm25_score(doc_id, query_tokens))
                  for doc_id in candidate_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def edit_distance(self, a, b):
        """Classic Levenshtein distance via dynamic programming."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # delete
                        dp[i][j - 1],      # insert
                        dp[i - 1][j - 1],  # substitute
                    )
        return dp[m][n]

    def correct_term(self, term, max_distance=2):
        """If term isn't in the index, find the closest indexed word within max_distance."""
        if term in self.inverted_index:
            return term
        best_word, best_dist = None, max_distance + 1
        for indexed_word in self.doc_freq:
            if abs(len(indexed_word) - len(term)) > max_distance:
                continue  # quick skip: length differs too much to be close
            dist = self.edit_distance(term, indexed_word)
            if dist < best_dist:
                best_word, best_dist = indexed_word, dist
        return best_word if best_word else term

    def hybrid_search(self, query, semantic, top_k=10, alpha=0.5):
        corrected_query = self.correct_query_text(query)
        bm25_results = dict(self.search(corrected_query, top_k=len(self.documents)))
        sem_scores = semantic.similarity_scores(corrected_query)
        max_bm25 = max(bm25_results.values()) if bm25_results else 1.0

        combined = []
        for doc_id in range(len(self.documents)):
            bm25_norm = bm25_results.get(doc_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            sem = float(sem_scores[doc_id])
            score = alpha * bm25_norm + (1 - alpha) * sem
            combined.append((doc_id, score, sem))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]   # now returns (doc_id, combined_score, raw_semantic_score)

    def add_paper_live(self, doc, jsonl_path="data/papers.jsonl"):
        """Add one new paper to the in-memory index AND persist it to disk."""
        self.add_document(doc)
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc) + "\n")
        return len(self.documents) - 1   # new doc's id

    def correct_word(self, w):
        if w in self.raw_vocab or w in STOPWORDS or len(w) <= 3 or w.isdigit():
            return w
        if w in self._corr_cache:
            return self._corr_cache[w]
        max_d = 1 if len(w) <= 5 else 2
        best, best_key = w, None
        for cand, freq in self.raw_vocab.items():
            if abs(len(cand) - len(w)) > max_d:
                continue
            d = self.edit_distance(w, cand)
            if d <= max_d:
                key = (d, -freq)          # closest first, then most common word
                if best_key is None or key < best_key:
                    best, best_key = cand, key
        self._corr_cache[w] = best
        return best

    def correct_query_text(self, query):
        return " ".join(self.correct_word(w) for w in TOKEN_RE.findall(query.lower()))

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2", emb_path="data/embeddings.npy"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.emb_path = emb_path
        self.vectors = np.load(emb_path)
        self._normalize()

    def _normalize(self):
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors = self.vectors / norms

    def query_vector(self, text):
        v = self.model.encode([text])[0]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def similarity_scores(self, query):
        q = self.query_vector(query)
        return self.vectors @ q

    def add_document(self, text):
        """Encode one new document and append it, keeping the .npy file in sync."""
        v = self.model.encode([text])[0]
        n = np.linalg.norm(v)
        v = v / n if n > 0 else v
        self.vectors = np.vstack([self.vectors, v])
        np.save(self.emb_path, self.vectors)

if __name__ == "__main__":
    idx = SearchIndex()
    idx.build("data/papers.jsonl")

    query = "graph neural network"
    print(f"\nSearching: '{query}'\n")
    results = idx.search(query, top_k=5)

    for rank, (doc_id, score) in enumerate(results, 1):
        doc = idx.documents[doc_id]
        print(f"{rank}. [{score:.2f}] {doc['title']}")