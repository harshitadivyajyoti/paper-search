import json
import re
import math
from collections import defaultdict

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "if", "than", "that", "this", "these", "those",
    "it", "its", "we", "our", "their", "which", "can", "using", "based",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    text = text.lower()
    tokens = TOKEN_RE.findall(text)          # keeps only letters/digits, splits on everything else
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class SearchIndex:
    def __init__(self):
        self.documents = []                   # list of {id, title, abstract}
        self.doc_tokens = []                  # tokenized tokens per doc, same order as documents
        self.inverted_index = defaultdict(set)  # word -> set of doc positions
        self.doc_freq = defaultdict(int)      # word -> number of docs containing it
        self.doc_len = []                     # token count per doc (needed for BM25 tomorrow)

    def add_document(self, doc):
        text = doc["title"] + " " + doc["abstract"]
        tokens = tokenize(text)
        doc_id = len(self.documents)

        self.documents.append(doc)
        self.doc_tokens.append(tokens)
        self.doc_len.append(len(tokens))

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
        print(f"Indexed {len(self.documents)} documents, "
              f"{len(self.inverted_index)} unique words")

    def candidates(self, query):
        """Return doc ids that contain at least one query word."""
        query_tokens = tokenize(query)
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
        avg_len = sum(self.doc_len) / len(self.doc_len)
        doc_len = self.doc_len[doc_id]
        tokens_in_doc = self.doc_tokens[doc_id]

        score = 0.0
        for term in query_tokens:
            freq = tokens_in_doc.count(term)     # how many times term appears in this doc
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

if __name__ == "__main__":
    idx = SearchIndex()
    idx.build("data/papers.jsonl")

    query = "graph neural network"
    print(f"\nSearching: '{query}'\n")
    results = idx.search(query, top_k=5)

    for rank, (doc_id, score) in enumerate(results, 1):
        doc = idx.documents[doc_id]
        print(f"{rank}. [{score:.2f}] {doc['title']}")