"""Control 4: tamper-evident, hash-chained JSONL audit log. Stores decisions only — never PII values or chunk text."""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass
from pathlib import Path

GENESIS = "0" * 64


@dataclass
class VerifyResult:
    ok: bool
    broken_at: int | None
    total: int


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(prev_hash: str, seq: int, ts: float, event: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical({"seq": seq, "ts": ts, "event": event})).encode()).hexdigest()


class HashChainedAuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def append(self, event: dict) -> dict:
        entries = self._entries()
        prev = entries[-1]["hash"] if entries else GENESIS
        seq, ts = len(entries) + 1, time.time()
        entry = {"seq": seq, "ts": ts, "event": event, "prev_hash": prev, "hash": _hash(prev, seq, ts, event)}
        with self.path.open("a") as f:
            f.write(_canonical(entry) + "\n")
        return entry

    def verify(self) -> VerifyResult:
        prev = GENESIS
        entries = self._entries()
        for i, e in enumerate(entries, start=1):
            if e["seq"] != i or e["prev_hash"] != prev or _hash(prev, e["seq"], e["ts"], e["event"]) != e["hash"]:
                return VerifyResult(False, i, len(entries))
            prev = e["hash"]
        return VerifyResult(True, None, len(entries))

    def tail(self, n: int = 20) -> list[dict]:
        return self._entries()[-n:]

    def tamper(self, seq: int) -> None:
        """Demo helper: mutate the stored event of entry `seq` without recomputing its hash."""
        lines = self.path.read_text().splitlines()
        e = json.loads(lines[seq - 1])
        e["event"] = {**e["event"], "tampered": True}
        lines[seq - 1] = _canonical(e)
        self.path.write_text("\n".join(lines) + "\n")

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
