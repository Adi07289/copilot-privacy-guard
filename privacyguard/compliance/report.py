"""Markdown/JSON renderers for the clause map, the DPIA register and the checklist."""
from __future__ import annotations
from privacyguard.compliance.checklist import CheckResult
from privacyguard.compliance.clauses import CLAUSES, CONTROL_NAMES
from privacyguard.compliance.dpia import RISKS, consistency_warnings, level


def _ev(store, keys) -> str:
    return "; ".join(f"{k.split(':', 1)[1]} = {store.fmt(k)}" for k in keys) or "—"


def render_compliance(store) -> str:
    lines = ["# Compliance map", "", "Paraphrases are ours, not legal text; each reference was verified (see *source*).", "",
             "| Framework | Ref | Obligation | Controls | Evidence (measured) | Note |", "|---|---|---|---|---|---|"]
    for c in CLAUSES:
        ctrls = ", ".join(CONTROL_NAMES[x] for x in c.controls)
        lines.append(f"| {c.framework} | {c.ref} | **{c.title}.** {c.obligation} | {ctrls} | {_ev(store, c.evidence)} | {c.note} |")
    lines += ["", "## Sources", ""] + [f"- {c.framework} {c.ref}: {c.source}" for c in CLAUSES]
    return "\n".join(lines) + "\n"


def render_dpia(store) -> str:
    lines = ["# DPIA risk register", "",
             "Likelihood (L) and impact (I) are on a 1–5 scale and are **expert judgement**; residual scores cite the "
             "measured evidence that justifies them.", "",
             "| ID | Threat | LINDDUN | OWASP | Inherent L×I | Controls | Residual L×I | Evidence | Rationale | Owner |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in RISKS:
        lines.append(f"| {r.id} | {r.threat} | {r.linddun} | {r.owasp} | {r.likelihood}×{r.impact} = {r.inherent} "
                     f"({level(r.inherent)}) | {', '.join(r.controls) or 'none'} | {r.residual_likelihood}×{r.residual_impact} "
                     f"= {r.residual} ({level(r.residual)}) | {_ev(store, r.evidence)} | {r.rationale} | {r.owner} |")
    warns = consistency_warnings(RISKS, store)
    lines += ["", "## Consistency check", ""] + ([f"- ⚠ {w}" for w in warns] or ["- No residual score contradicts the evidence."])
    return "\n".join(lines) + "\n"


def render_checklist(results: list[tuple]) -> str:
    tag = {True: "PASS", False: "FAIL", None: "NOT MEASURED"}
    passed = sum(1 for _, r in results if r.passed)
    lines = ["# Privacy Assurance Checklist", "", f"{passed}/{len(results)} checks pass. Every check runs against the "
             "system or its result files — none is self-declared.", "",
             "| ID | Statement | References | Result | Evidence |", "|---|---|---|---|---|"]
    for c, r in results:
        lines.append(f"| {c.id} | {c.statement} | {', '.join(c.refs)} | **{tag[r.passed]}** | {r.evidence} |")
    return "\n".join(lines) + "\n"


def export_json(store, results: list[tuple]) -> dict:
    return {"clauses": [{"framework": c.framework, "ref": c.ref, "title": c.title, "obligation": c.obligation,
                         "controls": list(c.controls), "evidence": {k: store.fmt(k) for k in c.evidence},
                         "note": c.note, "source": c.source} for c in CLAUSES],
            "risks": [{"id": r.id, "threat": r.threat, "linddun": r.linddun, "owasp": r.owasp,
                       "inherent": [r.likelihood, r.impact, r.inherent, level(r.inherent)],
                       "residual": [r.residual_likelihood, r.residual_impact, r.residual, level(r.residual)],
                       "controls": list(r.controls), "evidence": {k: store.fmt(k) for k in r.evidence},
                       "rationale": r.rationale, "owner": r.owner} for r in RISKS],
            "checks": [{"id": c.id, "statement": c.statement, "refs": list(c.refs), "passed": r.passed,
                        "evidence": r.evidence} for c, r in results],
            "dpia_warnings": consistency_warnings(RISKS, store)}
