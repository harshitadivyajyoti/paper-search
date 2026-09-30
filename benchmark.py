import time
from search_engine import SearchIndex

idx = SearchIndex()
build_start = time.perf_counter()
idx.build("data/papers.jsonl")
build_time = time.perf_counter() - build_start

queries = [
    "reinforcement learning",
    "neural network image classification",
    "database query optimization",
    "distributed systems consensus",
    "natural language processing",
]

times = []
for q in queries:
    start = time.perf_counter()
    results = idx.search(q, top_k=10)
    elapsed_ms = (time.perf_counter() - start) * 1000
    times.append(elapsed_ms)
    print(f"'{q}': {elapsed_ms:.2f} ms, top result: {results[0][1]:.2f} score" if results else f"'{q}': no results")

print(f"\nIndex build time: {build_time:.2f} s for {len(idx.documents)} documents")
print(f"Average query time: {sum(times)/len(times):.2f} ms")
print(f"Slowest query: {max(times):.2f} ms")