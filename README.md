# copilot-privacy-guard

Prototype of a **privacy-guard middleware** for retrieval-augmented enterprise LLM assistants (M365 Copilot-style),
built for *BCSE318L Data Privacy* (VIT Vellore, Fall 2026-27) — case study *"Privacy-Preserving Enterprise Copilots:
A DPDP-Aligned Risk Assessment and Mitigation Framework"*. This repo implements Review 2 §6.3 (Fig. 2) and the
first half of the Review 3 deliverables (O5 prototype, O6 evaluation).

```
User prompt ─► [1 PII redact] ─► Retriever (TF-IDF) ─► candidate chunks
   ─► [2 Purpose-based access filter] ─► [3 Injection screen] ─► [1 PII redact chunks] ─► LLM ─► answer
   every decision ─► [4 Hash-chained audit log]
```

`BaselineRAG` is the same retriever + LLM with **no** controls; `GuardedRAG` adds the four. Everything else is held
constant so the evaluation isolates the middleware.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pip install spacy && .venv/bin/python -m spacy download en_core_web_sm   # optional: NER for names
cp .env.example .env                       # add GROQ_API_KEY for real LLM answers; optional

.venv/bin/python scripts/generate_corpus.py          # data/corpus.json  (80 docs, 369 PII spans, 4 poisoned)
.venv/bin/python scripts/run_eval.py                 # results/eval.md + eval.json
.venv/bin/streamlit run app/streamlit_app.py         # side-by-side demo at http://localhost:8501
.venv/bin/pytest -q                                  # 46 tests
.venv/bin/python scripts/tamper_demo.py              # flip one audit line, watch verify() fail
```

## Current results (seed 42, extractive provider, spaCy names)

| Metric | Baseline RAG | Guarded RAG | Status |
|---|---|---|---|
| PII leakage rate | 34.8% | **0.0%** | measured |
| Over-sharing retrieval rate | 60.8% | **0.0%** | measured |
| Redaction precision / recall | — | 92.9% / 95.9% | measured |
| Added latency | 0 | +22 ms | measured |
| Prompt-injection success rate | — | — | next stage (needs LLM-judge + generation eval) |
| Answer-quality retention | — | — | next stage |

## Layout

| Path | Responsibility |
|---|---|
| `privacyguard/models.py` | `User`, `Document`, `Chunk`, `Decision`, `GuardContext` |
| `privacyguard/corpus/indian_pii.py` | Aadhaar (Verhoeff), PAN, Indian mobile, IFSC, UPI VPA, card (Luhn) — generators + validators |
| `privacyguard/corpus/generator.py` | Seeded Faker(`en_IN`) corpus: emails / HR records / tickets / code, tagged with department · sensitivity · purposes, ground-truth PII spans, 4 poisoned docs |
| `privacyguard/retrieval.py` | `Retriever` protocol; `TfidfIndex` |
| `privacyguard/llm.py` | `GroqProvider` (OpenAI-compatible) with `ExtractiveProvider` fallback |
| `privacyguard/controls/pii.py` | Control 1 — India-first recognizer registry + optional spaCy NER; typed placeholders `<AADHAAR_1>` |
| `privacyguard/controls/access.py` | Control 2 — clearance ≥ sensitivity, department match, declared purpose ∈ doc purposes |
| `privacyguard/controls/injection.py` | Control 3 — weighted pattern bank + zero-width / HTML-comment / base64 checks; `LLMJudge` hook (stubbed) |
| `privacyguard/controls/audit.py` | Control 4 — SHA-256 hash-chained JSONL; `verify()` reports first broken link |
| `privacyguard/pipeline.py` | `BaselineRAG`, `GuardedRAG`, `build_default()` |
| `privacyguard/eval/` | 20-query set, metric functions, harness → `results/eval.{md,json}` |
| `app/streamlit_app.py` | Compare · Audit log · Evaluation tabs |
| `docs/` | design spec, implementation plan, `ROADMAP.md`, `PRESENTATION_SCRIPT.md` |

## Known limitations (stated, not hidden)

- Retrieval is sparse TF-IDF; dense embeddings and embedding-inversion defenses are not implemented.
- Injection screening is heuristic; the LLM-as-judge pass is a stub. Adversarial paraphrases will evade it.
- The corpus is synthetic. PIIBench shows detectors generalise poorly to real mixed enterprise text; expect lower P/R there.
- The access policy is a static tag model, not an identity-provider integration.
- Sensitive-attribute inference and fine-tuning memorisation are in the threat taxonomy but have no control here.
- The audit log is tamper-*evident*, not tamper-*proof*: it detects modification, it does not prevent it.
