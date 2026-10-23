# Presentation script — Final review

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

**5 · Tech stack (30 s).** One Python package, twelve configurations of one pipeline, 123 tests. Groq runs qwen/qwen3.8-27b for answers and openai/gpt-oss-120b as the judge; Presidio and Meta's Prompt Guard 2 are comparators. All LLM outputs are cached and committed.

**6 · Objectives (30 s).** All ten delivered: incidents, the DPIA register, a 24-clause compliance map and a 13-check assurance checklist joined the prototype and the evaluation. Literature grew from 14 to 32 verified papers.

**7 · Literature positioning (60 s).** Every defence we found covers one column — decoding noise, delimiting, content classifiers, PII redaction, embedding noise. None knows who is asking or why, none protects the index, audits and is measured end to end, and none is DPDP-mapped. That row of ticks at the bottom is our contribution.

**8 · Setup (40 s).** 80 synthetic Indian-enterprise documents with 369 labelled PII spans and 4 poisoned documents; 36 queries from five personas. Every defence is a configuration of the same pipeline — same corpus, retriever and LLM — so only the defence changes. Injection queries come from a persona *allowed* to read the poisoned document, so we test the injection control, not access control. Non-ours rows are simplified analogues, not reproductions.

**9 · Headline results (60 s).** PII sent to the provider: 100% → 0%. Over-sharing: 50.9% → 0%. Raw PII in the index and poisoned chunks reaching the model: both to zero. The honest rows: answer-quality retention falls to 34.9% ROUGE-L, median added latency is +97 ms, and an uncached judge call takes about 553 ms. The model itself obeyed almost no canaries — vanilla 0% of 8 — but that is model behaviour, not a control we own, so we design for zero exposure.

**10 · Comparative study (75 s).** Read it by column. Copilot-style RBAC cuts over-sharing to 0.6% but still sends 49% of PII to the provider and lets 59% of poisoned chunks through. Redaction-only leaves over-sharing at 51%. Output filtering looks clean in the answer while the provider still receives 100% — under DPDP §8(1) that is still the fiduciary's responsibility. Every single-family defence leaves an exposure open; ours closes all four.

**11 · Ablation (40 s).** Remove one control and exactly one exposure returns: no access filter → over-sharing 55%; no redaction → 49% of PII to the provider; no redacted index → the index holds every value again. Removing the judge changes nothing on the corpus but drops latency to +98 ms — its value is on harder payloads.

**12 · PII vs Presidio (50 s).** In Review 3 we called Presidio US-centric; we checked — it ships Aadhaar and PAN recognizers, disabled. So we measured: held-out F1 89.5% for ours vs 54.3% as shipped and 64.1% with India recognizers on. Presidio has nothing for IFSC, UPI or rupee salaries — and it beats us on spaced mobiles and written dates, which are our gaps.

**13 · Injection detection (50 s).** Patterns catch 33% of 24 payloads — none paraphrased or Hindi — and flag 62% of hard negatives. The LLM judge catches 100% with 0% false positives. Meta's Prompt Guard 2 catches 33%. Lesson: OR-ing in the heuristic inherits its false positives — the judge should decide alone.

**14 · Compliance (45 s).** Compliance as code: 24 clauses across DPDP Act, DPDP Rules 2025, GDPR, EU AI Act and OWASP, each verified against source text and pointing at a measured number; a 7-risk DPIA whose residual scores cite evidence; and 13 checks run against the system — 12 pass. A12, utility, fails, and we left it failing.

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
- **0% looks too good.** It is 0% on 36 queries; the held-out PII F1 (89.5%) shows the detector isn't perfect. Real text would leak some.
- **Answer quality is low — is it usable?** 34.9% ROUGE-L is the measured cost; much of it is names/contacts the guard deliberately removes. Future work re-identifies them for authorised users.
- **What does the judge cost?** median +97 ms per query with caches; each uncached judge call ≈ 553 ms. Batching reduces it.
- **Which DPDP sections?** §4 and §6(1) purpose limitation → access filter; §8(1) processors and §8(5) safeguards → redaction, redacted index, screening; §8(6) and Rules 6–7 → audit chain; §10(2)(c) and Rule 13 → DPIA + checklist.
- **Why did the LLM resist the canaries?** Modern models are trained against obvious injections; we measured it and report it. The guard removes the payload so safety doesn't depend on it.
- **Is the corpus real?** Synthetic, seeded and labelled — so every number reproduces. A held-out set written in different formats limits bias.
- **Two detector fixes after the first run?** Sign-off first names and names crossing line breaks into titles; both general, disclosed in the report; the held-out set contains neither.
- **What if Groq is down?** Cached results replay; uncached calls fall back to an extractive answerer and the judge falls back to the heuristic — recorded per call.
- **How is the audit log protected?** Each entry hashes the previous one; any edit or deletion breaks verification at that entry. Tamper-evident, not tamper-proof.
