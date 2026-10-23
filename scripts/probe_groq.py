# scripts/probe_groq.py
"""Print each candidate model's rate limits (from response headers) without printing the key."""
import json, os, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
SKIP = ("whisper", "tts", "orpheus", "playai", "compound", "prompt-guard", "safeguard")

if __name__ == "__main__":
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=0)
    rows = []
    for m in sorted(x.id for x in client.models.list().data if not any(k in x.id for k in SKIP)):
        t0 = time.perf_counter()
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=m, max_tokens=300, temperature=0, messages=[{"role": "user", "content": "Reply with OK."}])
            h = raw.headers
            rows.append({"model": m, "ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000),
                         "rpm": h.get("x-ratelimit-limit-requests"), "tpm": h.get("x-ratelimit-limit-tokens"),
                         "reply": raw.parse().choices[0].message.content})
        except Exception as e:
            rows.append({"model": m, "ok": False, "error": type(e).__name__ + ": " + str(e)[:120]})
    for r in rows:
        print(r)
    Path(ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "groq_probe.json").write_text(json.dumps(rows, indent=1))
