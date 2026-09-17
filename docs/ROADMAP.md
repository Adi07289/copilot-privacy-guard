# Roadmap — done vs. next stage

Status legend: ✅ delivered · 🟨 built tonight (this repo) · ⬜ next stage

## Objectives (from Review 1, tracked through Review 2)

| # | Objective | Review 2 status | Now | Evidence |
|---|---|---|---|---|
| O1 | Review 10–15 works on memorisation, MIA, RAG leakage, injection, embedding inversion | ✅ 14-paper matrix | ✅ | Review 2 §4 |
| O2 | Data-flow map of a RAG enterprise assistant | ✅ Fig. 1 | ✅ | Review 2 §6.1 |
| O3 | Threat taxonomy → LINDDUN + OWASP LLM Top 10 | ✅ | ✅ | Review 2 §6.2 |
| O4 | ≥2 real-world incident analyses | in progress | ⬜ | — |
| O5 | Prototype privacy-guard middleware | design only (Fig. 2) | 🟨 **~50%** | `privacyguard/`, 46 tests, Streamlit demo |
| O6 | Quantitative evaluation protocol applied | defined | 🟨 **4 of 6 metrics measured** | `results/eval.md` |
| O7 | DPIA-style risk register | in progress | ⬜ | — |
| O8 | Clause mapping: DPDP 2023 / Rules 2025 / GDPR / EU AI Act | in progress | ⬜ | — |
| O9 | Privacy Assurance Checklist | final review | ⬜ | — |
| O10 | Limitations + future work | final review | 🟨 drafted | `README.md`, `PRESENTATION_SCRIPT.md` |

## The four controls (Fig. 2)

| Control | Built | Not yet |
|---|---|---|
| 1. PII detection & redaction | 🟨 India-first recognizer registry (Aadhaar/Verhoeff, PAN, mobile, IFSC, UPI, email, card/Luhn, DOB, ₹ salary) + spaCy PERSON NER; typed placeholders; applied to prompt **and** retrieved chunks | ⬜ Presidio adapter for parity comparison; address/passport/voter-ID recognizers; span-level P/R on a non-synthetic set |
| 2. Purpose-based access filter | 🟨 clearance ≥ sensitivity → department (or public / CISO-DPO cross-dept) → declared purpose ∈ doc purposes; reason recorded per drop | ⬜ integration with a real identity provider (Entra/Graph); per-document ACL import; purpose declared via UI consent prompt |
| 3. Prompt-injection screening | 🟨 weighted pattern bank + zero-width / HTML-comment-imperative / base64-decoded checks; below-threshold chunks wrapped in `<untrusted_document>` (StruQ-style separation); `LLMJudge` protocol present | ⬜ LLM-as-judge implementation (Groq); adversarial paraphrase test set; measured injection-success rate |
| 4. Tamper-evident audit | 🟨 SHA-256 hash chain over canonical JSON; `verify()` reports first broken seq; stores entity **types/counts** only, never values | ⬜ periodic anchor of the head hash to an external store; log rotation with chained segments |

## Six metrics (Review 2 §6.4)

| Metric | Status | Baseline → Guarded (seed 42) |
|---|---|---|
| PII leakage rate | 🟨 measured | 34.8% → 0.0% |
| Over-sharing retrieval rate | 🟨 measured | 60.8% → 0.0% |
| Redaction precision / recall | 🟨 measured | 92.9% / 95.9% |
| Added latency | 🟨 measured | +22 ms |
| Prompt-injection success rate | ⬜ needs generation-level judging | — |
| Answer-quality retention (ROUGE-L / cosine) | ⬜ needs Groq runs on non-sensitive queries | — |

## Next-stage build order (Review 3 → final)

1. **Answer-quality retention** — run the 11 non-sensitive queries through Groq for both pipelines, ROUGE-L + cosine. (Closes O6.)
2. **LLM-as-judge** — `GroqJudge.judge(text) -> float`; injection-success rate = seeded instruction followed in the *generated* answer. (Closes control 3 + O6.)
3. **DPDP clause mapping** — table: control → DPDP §4/§6/§8/§9, Rules 2025 r.6/r.7, GDPR Art.5/25/32, EU AI Act Art.9/10. (O8.)
4. **DPIA risk register** — 6 threats × likelihood × impact → inherent risk → control → residual. (O7.)
5. **Incident analyses** — two documented Copilot / enterprise-assistant disclosures with root-cause chain. (O4.)
6. **Privacy Assurance Checklist** — prioritised, verifiable statements derived from 1–5. (O9.)
7. Dense-embedding retriever behind the existing `Retriever` protocol; SPARSE-style concept-noise defence experiment.
