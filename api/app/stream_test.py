import time
import httpx

url = "http://localhost:8000/chat/stream"
body = {"question": "what is your return policy?", "user_id": "1", "session_id": "test"}

start = time.perf_counter()
with httpx.stream("POST", url, json=body, timeout=120) as r:
    print("status:", r.status_code, r.headers["content-type"])
    for line in r.iter_lines():
        if line.startswith("event:") or line.startswith("data:"):
            print(f"{time.perf_counter() - start:6.2f}s  {line}")
            