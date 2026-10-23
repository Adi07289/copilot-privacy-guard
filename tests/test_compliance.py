import json
from privacyguard.compliance import checklist as CK
from privacyguard.compliance.clauses import CLAUSES, CONTROL_IDS
from privacyguard.compliance.dpia import RISKS, Risk, consistency_warnings, level
from privacyguard.compliance.evidence import EvidenceStore
from privacyguard.compliance.report import render_checklist, render_compliance, render_dpia
from privacyguard.configs import CONFIG_ORDER
from privacyguard.controls.audit import HashChainedAuditLog
from privacyguard.eval.harness import METRICS

FRAMEWORKS = {"DPDP Act 2023", "DPDP Rules 2025", "GDPR", "EU AI Act", "OWASP LLM Top 10 (2025)"}

def _store(value=0.0):
    agg = {c: {k: value for k, _, _ in METRICS} for c in CONFIG_ORDER}
    det = {"heuristic": 0.5, "judge": 0.9, "combined": 0.95}
    bench = {"tpr": det, "fpr_all": det, "fpr_clean": det, "fpr_hard": det}
    pii = {"results": {ds: {d: {"precision": 0.9, "recall": 0.9, "f1": 0.9} for d in ("ours", "presidio_default", "presidio_india")}
                       for ds in ("corpus", "heldout")}}
    s = EvidenceStore(agg, bench, pii)
    s.checks = {c.id: CK.CheckResult(True, "ok") for c in CK.CHECKS}
    return s

def test_clauses_well_formed():
    assert len(CLAUSES) >= 15
    for c in CLAUSES:
        assert c.framework in FRAMEWORKS and set(c.controls) <= set(CONTROL_IDS) and c.evidence and c.source.startswith("http")

def test_every_evidence_key_resolves():
    s = _store()
    for key in [e for c in CLAUSES for e in c.evidence] + [e for r in RISKS for e in r.evidence]:
        s.fmt(key)                                                    # raises KeyError if unresolvable

def test_dpia_math():
    assert level(4) == "Low" and level(9) == "Medium" and level(16) == "High" and level(20) == "Critical"
    assert all(r.residual <= r.inherent for r in RISKS) and len(RISKS) == 7

def test_consistency_warnings_flag_optimistic_residuals():
    r = Risk("RX", "t", "l", "o", 4, 4, ("access",), 1, 4, ("metric:full.oversharing_rate",), "why", "CISO")
    assert consistency_warnings([r], _store(0.0)) == []
    assert consistency_warnings([r], _store(0.2))

def test_renderers_produce_tables():
    s = _store()
    assert "| DPDP Act 2023 |" in render_compliance(s)
    assert "expert judgement" in render_dpia(s)
    assert "PASS" in render_checklist([(c, CK.CheckResult(True, "ok")) for c in CK.CHECKS])

def test_audit_checks_pass_then_fail(tmp_path, mini_corpus):
    p = tmp_path / "a.jsonl"
    log = HashChainedAuditLog(p)
    log.append({"user": "u-eng", "purpose": "engineering", "control": "access", "action": "drop", "doc_id": "hr-1",
                "detail": {"reason": "clearance"}})
    ctx = CK.CheckContext(_store(), mini_corpus, p)
    res = {c.id: r for c, r in CK.run_checks(ctx)}
    assert res["A07"].passed and res["A08"].passed and res["A09"].passed and res["A10"].passed
    log.append({"user": "u-eng", "purpose": "engineering", "control": "pii", "action": "x", "doc_id": None,
                "detail": {"leak": mini_corpus[0].pii_spans[0].value}})
    log.tamper(1)
    res = {c.id: r for c, r in CK.run_checks(ctx)}
    assert res["A08"].passed is False and res["A09"].passed is False

def test_offline_check_passes():
    res = {c.id: r for c, r in CK.run_checks(CK.CheckContext(_store(), [], None))}
    assert res["A13"].passed is True
