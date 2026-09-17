"""Usage: .venv/bin/python scripts/tamper_demo.py [--path data/audit.jsonl] [--seq N]"""
import argparse
from privacyguard.controls.audit import HashChainedAuditLog

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/audit.jsonl")
    ap.add_argument("--seq", type=int, default=None)
    a = ap.parse_args()
    log = HashChainedAuditLog(a.path)
    before = log.verify()
    print(f"before: ok={before.ok} total={before.total}")
    if before.total == 0:
        raise SystemExit("audit log is empty — run a guarded query first")
    seq = a.seq or max(1, before.total // 2)
    log.tamper(seq)
    after = log.verify()
    print(f"tampered entry {seq} → verify: ok={after.ok} broken_at={after.broken_at}")
