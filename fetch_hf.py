import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

DATASET = "gfissore/arxiv-abstracts-2021"
TARGET = 5000             # CS papers we want
LENGTH = 100
DATA_FILE = "data/papers.jsonl"
OFFSET_FILE = "data/offset.txt"
PAUSE = 2                 # seconds between requests (stay under the rate limit)
HEADERS = {"User-Agent": "paper-search-student-project"}


def get_rows(offset):
    params = urllib.parse.urlencode({
        "dataset": DATASET, "config": "default", "split": "train",
        "offset": offset, "length": LENGTH,
    })
    url = "https://datasets-server.huggingface.co/rows?" + params
    for attempt in range(1, 8):
        wait = 20 * attempt
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = e.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after) + 2
                print(f"  rate limited, waiting {wait}s...")
            else:
                print(f"  HTTP {e.code}, waiting {wait}s...")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  {e}, waiting {wait}s...")
        time.sleep(wait)
    raise RuntimeError("Still failing. Wait 10 minutes and rerun; progress is saved.")


def clean(text):
    return " ".join((text or "").split())


def count_saved():
    if not os.path.exists(DATA_FILE):
        return 0
    with open(DATA_FILE, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    os.makedirs("data", exist_ok=True)
    saved = count_saved()
    if os.path.exists(OFFSET_FILE):
        offset = int(open(OFFSET_FILE).read())
    else:
        offset = 3000 if saved else 0      # the earlier run scanned 3000 rows
    print(f"Resuming at row {offset} with {saved} papers saved.")

    while saved < TARGET:
        data = get_rows(offset)
        rows = data.get("rows", [])
        if not rows:
            break
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            for item in rows:
                r = item["row"]
                cats = str(r.get("categories") or "")
                if "cs." not in cats:
                    continue
                title, abstract = clean(r.get("title")), clean(r.get("abstract"))
                if not title or not abstract:
                    continue
                f.write(json.dumps({
                    "id": r.get("id"), "title": title, "abstract": abstract,
                    "authors": clean(r.get("authors")), "categories": cats,
                }) + "\n")
                saved += 1
        offset += LENGTH
        with open(OFFSET_FILE, "w") as f:
            f.write(str(offset))
        if (offset // LENGTH) % 10 == 0:
            print(f"scanned {offset} rows, saved {saved} CS papers")
        time.sleep(PAUSE)

    print(f"Done. {saved} papers in {DATA_FILE}")


if __name__ == "__main__":
    main()