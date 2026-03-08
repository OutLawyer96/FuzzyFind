import requests
import json

BASE = "http://localhost:8000"


def show(label, r):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(json.dumps(r.json(), indent=2))


# 1. health check
show("HEALTH CHECK", requests.get(f"{BASE}/health"))

# 2. first query - cache miss
show("QUERY 1 — cache miss", requests.post(f"{BASE}/query",
    json={"query": "best graphics cards for gaming"}))

# 3. same query - cache hit
show("QUERY 2 — same query, expect cache hit", requests.post(f"{BASE}/query",
    json={"query": "best graphics cards for gaming"}))

# 4. different topic - should miss
show("QUERY 3 — different topic, expect miss", requests.post(f"{BASE}/query",
    json={"query": "middle east conflict and war"}))

# 5. stats
show("CACHE STATS", requests.get(f"{BASE}/cache/stats"))

# 6. flush
show("FLUSH CACHE", requests.delete(f"{BASE}/cache"))

# 7. confirm empty
show("STATS AFTER FLUSH", requests.get(f"{BASE}/cache/stats"))
