import json
from privacyguard.controls.audit import HashChainedAuditLog, GENESIS

def test_append_and_verify(tmp_path):
    log = HashChainedAuditLog(tmp_path / "a.jsonl")
    e1 = log.append({"control": "pii", "action": "redact_query"})
    e2 = log.append({"control": "access", "action": "drop", "doc_id": "hr-001"})
    assert e1["prev_hash"] == GENESIS and e2["prev_hash"] == e1["hash"] and e2["seq"] == 2
    v = log.verify()
    assert v.ok and v.total == 2 and v.broken_at is None

def test_empty_log_verifies(tmp_path):
    v = HashChainedAuditLog(tmp_path / "none.jsonl").verify()
    assert v.ok and v.total == 0

def test_tamper_payload_detected(tmp_path):
    log = HashChainedAuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.append({"i": i})
    log.tamper(2)
    v = log.verify()
    assert not v.ok and v.broken_at == 2

def test_deleting_middle_line_detected(tmp_path):
    p = tmp_path / "a.jsonl"
    log = HashChainedAuditLog(p)
    for i in range(3):
        log.append({"i": i})
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")
    v = log.verify()
    assert not v.ok and v.broken_at == 2

def test_tail(tmp_path):
    log = HashChainedAuditLog(tmp_path / "a.jsonl")
    for i in range(5):
        log.append({"i": i})
    assert [e["event"]["i"] for e in log.tail(2)] == [3, 4]
