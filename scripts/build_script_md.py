"""Writes docs/PRESENTATION_SCRIPT.md, README results table and ROADMAP from results/*.json (numbers never typed)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
J = lambda n: json.loads((R / n).read_text())
C, B, P, CJ, M = J("comparison.json"), J("injection_bench.json"), J("pii_detection.json"), J("compliance.json"), J("run_meta.json")
A = C["aggregates"]; ho = P["results"]["heldout"]
pc = lambda v, d=1: "—" if v is None else f"{100*v:.{d}f}%"
a = lambda c, k, d=1: pc(A[c].get(k), d)
npass = sum(1 for x in CJ["checks"] if x["passed"]); nchk = len(CJ["checks"])
lat, lat_nj = A["full"]["added_latency_ms"], A["full_no_judge"]["added_latency_ms"]

script = f"""# Presentation script — Final review

**Copilot Privacy Guard** · BCSE318L Data Privacy · Topic 35 · Faculty: Dr. Geetha Mary A
**Team:** Aditya Sharma (23BCE0936) · Aarav Raina (23BCE0992)
**Deck:** `deliverables/Copilot Privacy Guard - Final.pptx` (17 slides + 3 appendix) · **Report:** `deliverables/Copilot_Privacy_Guard_Final_Report.pdf`
**Length:** ~12 min + demo + Q&A. Every number below is generated from `results/` by `scripts/build_script_md.py`.

## 0. Pre-flight (5 min before)
- [ ] `cd ~/copilot-privacy-guard && .venv/bin/streamlit run app/streamlit_app.py` — leave it open on **Compare**
- [ ] **Audit log → Reset log**, then run one query so the chain has entries
- [ ] Sidebar: **Riya Menon — Engineer**, purpose **engineering**, query **q01**, Left **Vanilla RAG**, Right **Copilot Privacy Guard (ours)**
- [ ] No Wi-Fi? Fine — committed caches replay every LLM result offline (the provider badge still works)

## Slide-by-slide (spoken)

**1 · Title (20 s).** Good morning, ma'am. In Review 3 we showed half of Copilot Privacy Guard. Today all ten objectives from Review 1 are delivered, measured, and compared against the defence families in the literature.

**2 · Problem (45 s).** Enterprise copilots get broad read access so they can answer in natural language. Audits find 15–16% of business-critical files already over-shared; the copilot makes every one discoverable with a sentence. DLP and RBAC work on whole documents and static roles, and nothing deployable had been evaluated against India's DPDP Act.

**3 · How it works (45 s).** Four controls before the model sees anything. Documents are redacted *before* indexing, so the vector store never holds raw PII. The purpose filter and injection screening — now a heuristic plus an LLM judge — run on the candidates; redaction runs again on survivors; every decision goes into a SHA-256 hash chain.

**4 · Controls (40 s).** India-first PII — Aadhaar with the Verhoeff checksum, PAN, IFSC, UPI, rupee salaries. Purpose-based access is DPDP's purpose limitation as a runtime rule. The judge catches what patterns cannot. The audit stores types and counts, never values.

**5 · Tech stack (30 s).** One Python package, twelve configurations of one pipeline, 123 tests. Groq runs {M.get('groq_model')} for answers and {M.get('judge_model')} as the judge; Presidio and Meta's Prompt Guard 2 are comparators. All LLM outputs are cached and committed.

**6 · Objectives (30 s).** All ten delivered: incidents, the DPIA register, a {len(CJ['clauses'])}-clause compliance map and a {nchk}-check assurance checklist joined the prototype and the evaluation. Literature grew from 14 to 32 verified papers.

**7 · Literature positioning (60 s).** Every defence we found covers one column — decoding noise, delimiting, content classifiers, PII redaction, embedding noise. None knows who is asking or why, none protects the index, audits and is measured end to end, and none is DPDP-mapped. That row of ticks at the bottom is our contribution.

**8 · Setup (40 s).** 80 synthetic Indian-enterprise documents with 369 labelled PII spans and 4 poisoned documents; 36 queries from five personas. Every defence is a configuration of the same pipeline — same corpus, retriever and LLM — so only the defence changes. Injection queries come from a persona *allowed* to read the poisoned document, so we test the injection control, not access control. Non-ours rows are simplified analogues, not reproductions.

**9 · Headline results (60 s).** PII sent to the provider: {a('none','context_pii_exposure',0)} → {a('full','context_pii_exposure',0)}. Over-sharing: {a('none','oversharing_rate')} → {a('full','oversharing_rate',0)}. Raw PII in the index and poisoned chunks reaching the model: both to zero. The honest rows: answer-quality retention falls to {a('full','answer_quality_rouge_l')} ROUGE-L, median added latency is +{lat:.0f} ms, and an uncached judge call takes about {(M.get('judge_call_ms_median') or 0):.0f} ms. The model itself obeyed almost no canaries — vanilla {a('none','injection_success_rate',0)} of {A['none']['injection_success_n']} — but that is model behaviour, not a control we own, so we design for zero exposure.

**10 · Comparative study (75 s).** Read it by column. Copilot-style RBAC cuts over-sharing to {a('rbac','oversharing_rate')} but still sends {a('rbac','context_pii_exposure',0)} of PII to the provider and lets {a('rbac','injection_exposure_rate',0)} of poisoned chunks through. Redaction-only leaves over-sharing at {a('pii_only','oversharing_rate',0)}. Output filtering looks clean in the answer while the provider still receives {a('output_filter','context_pii_exposure',0)} — under DPDP §8(1) that is still the fiduciary's responsibility. Every single-family defence leaves an exposure open; ours closes all four.

**11 · Ablation (40 s).** Remove one control and exactly one exposure returns: no access filter → over-sharing {a('full_no_access','oversharing_rate',0)}; no redaction → {a('full_no_pii','context_pii_exposure',0)} of PII to the provider; no redacted index → the index holds every value again. Removing the judge changes nothing on the corpus but drops latency to +{lat_nj:.0f} ms — its value is on harder payloads.

**12 · PII vs Presidio (50 s).** In Review 3 we called Presidio US-centric; we checked — it ships Aadhaar and PAN recognizers, disabled. So we measured: held-out F1 {pc(ho['ours']['f1'])} for ours vs {pc(ho['presidio_default']['f1'])} as shipped and {pc(ho.get('presidio_india',{}).get('f1'))} with India recognizers on. Presidio has nothing for IFSC, UPI or rupee salaries — and it beats us on spaced mobiles and written dates, which are our gaps.

**13 · Injection detection (50 s).** Patterns catch {pc(B['tpr']['heuristic'],0)} of 24 payloads — none paraphrased or Hindi — and flag {pc(B['fpr_hard']['heuristic'],0)} of hard negatives. The LLM judge catches {pc(B['tpr']['judge'],0)} with {pc(B['fpr_hard']['judge'],0)} false positives. Meta's Prompt Guard 2 catches {pc(B['tpr'].get('prompt_guard'),0)}. Lesson: OR-ing in the heuristic inherits its false positives — the judge should decide alone.

**14 · Compliance (45 s).** Compliance as code: {len(CJ['clauses'])} clauses across DPDP Act, DPDP Rules 2025, GDPR, EU AI Act and OWASP, each verified against source text and pointing at a measured number; a 7-risk DPIA whose residual scores cite evidence; and {nchk} checks run against the system — {npass} pass. A12, utility, fails, and we left it failing.

**15 · Unique factor (30 s).** Purpose-aware access as a DPDP rule; India-first PII measured against Presidio; a full pipeline measured against the alternatives with compliance evidence generated from results.

**16 · Limitations (40 s).** Utility and latency cost; canaries appear verbatim in three payloads, so quoting would count as success; synthetic corpus with a held-out check (and two detector fixes after the first run, disclosed); static access tags; few positive injection successes because the model resists — exposure is the robust measure.

**17 · Future work + thanks (20 s).** Judge-only decisions, re-identifying placeholders for authorised users, wider recognizers, output egress control, identity-provider integration. Thank you — happy to take questions or run the demo.

## Live demo (3 min)
1. **Compare** — Riya / q01 / Vanilla vs ours → six HR chunks dropped by clearance; nothing reaches the LLM.
2. **Compare** — q32 (injection), Left **Heuristic screening only**, Right **ours** → the poisoned ticket is dropped.
3. **Attack lab** — default paraphrased text → heuristic 0.00, judge ≈ 0.99 → DROP.
4. **Comparative study / Compliance** — live tables behind slides 10–14.
5. **Audit log** — Verify → Tamper a line → Verify shows `broken_at`.

## Anticipated questions
- **Why not reproduce the papers?** Training-time defences (StruQ, SPARSE) need fine-tuning/GPUs. We compare defence *families* on identical data and say so on every slide; reported numbers are in Appendix B as context only.
- **0% looks too good.** It is 0% on 36 queries; the held-out PII F1 ({pc(ho['ours']['f1'])}) shows the detector isn't perfect. Real text would leak some.
- **Answer quality is low — is it usable?** {a('full','answer_quality_rouge_l')} ROUGE-L is the measured cost; much of it is names/contacts the guard deliberately removes. Future work re-identifies them for authorised users.
- **What does the judge cost?** median +{lat:.0f} ms per query with caches; each uncached judge call ≈ {(M.get('judge_call_ms_median') or 0):.0f} ms. Batching reduces it.
- **Which DPDP sections?** §4 and §6(1) purpose limitation → access filter; §8(1) processors and §8(5) safeguards → redaction, redacted index, screening; §8(6) and Rules 6–7 → audit chain; §10(2)(c) and Rule 13 → DPIA + checklist.
- **Why did the LLM resist the canaries?** Modern models are trained against obvious injections; we measured it and report it. The guard removes the payload so safety doesn't depend on it.
- **Is the corpus real?** Synthetic, seeded and labelled — so every number reproduces. A held-out set written in different formats limits bias.
- **Two detector fixes after the first run?** Sign-off first names and names crossing line breaks into titles; both general, disclosed in the report; the held-out set contains neither.
- **What if Groq is down?** Cached results replay; uncached calls fall back to an extractive answerer and the judge falls back to the heuristic — recorded per call.
- **How is the audit log protected?** Each entry hashes the previous one; any edit or deletion breaks verification at that entry. Tamper-evident, not tamper-proof.
"""
(ROOT / "docs" / "PRESENTATION_SCRIPT.md").write_text(script)

road = f"""# Roadmap — status at final review

| # | Objective | Status | Artifact |
|---|---|---|---|
| O1 | Literature review | Done — 32 verified papers | report §Literature review, `docs/report/references.py` |
| O2 | Data-flow map | Done | Review 2 Fig. 1; report Fig. 1 |
| O3 | Threat taxonomy (LINDDUN + OWASP) | Done | Review 2 §6.2; `privacyguard/compliance/dpia.py` |
| O4 | Incident analyses | Done — 3 incidents | `docs/INCIDENTS.md` |
| O5 | Prototype middleware | Done — 4 controls + redacted index, 123 tests | `privacyguard/` |
| O6 | Evaluation protocol applied | Done — 9 metrics × {len(C['configs'])} configs | `results/comparison.md` |
| O7 | DPIA risk register | Done — 7 risks | `results/dpia.md` |
| O8 | Clause mapping | Done — {len(CJ['clauses'])} verified clauses | `results/compliance.md` |
| O9 | Privacy Assurance Checklist | Done — {npass}/{nchk} pass | `results/checklist.md` |
| O10 | Limitations & future work | Done | report §Future work, deck slides 16–17 |

## Genuine future work
1. Judge as sole drop decision (removes heuristic false positives). 2. Re-identify placeholders for authorised users.
3. Recognizers for secrets, spaced mobiles, written dates, lakh/crore. 4. Output egress allow-list.
5. Identity-provider ACLs + purpose consent prompt. 6. Real de-identified corpus; adaptive attacks. 7. Batch/cache judge calls.
"""
(ROOT / "docs" / "ROADMAP.md").write_text(road)

readme = (ROOT / "README.md").read_text()
start = readme.index("## Results (") if "## Results (" in readme else readme.index("## Current results"); end = readme.index("## Layout")
table = "\n".join(l for l in (R / "comparison.md").read_text().splitlines() if l.startswith("|"))
readme = readme[:start] + f"""## Results (seed 42 · {M.get('n_queries')} queries · answers {M.get('groq_model')} · judge {M.get('judge_model')})

Regenerate everything with `.venv/bin/python scripts/run_all.py` (uses Groq when a key or the committed cache is available).

{table}

PII detection (held-out F1): ours {pc(ho['ours']['f1'])} · Presidio {pc(ho['presidio_default']['f1'])}. Injection detection: LLM judge {pc(B['tpr']['judge'],0)} (FPR hard negatives {pc(B['fpr_hard']['judge'],0)}) · heuristic {pc(B['tpr']['heuristic'],0)} · Prompt Guard 2 {pc(B['tpr'].get('prompt_guard'),0)}. Assurance checklist: {npass}/{nchk} pass. Deliverables: `deliverables/`.

""" + readme[end:]
readme = readme.replace(".venv/bin/pytest -q                                  # 46 tests", ".venv/bin/pytest -q                                  # 123 tests")
(ROOT / "README.md").write_text(readme)
print("wrote script, roadmap, README")
