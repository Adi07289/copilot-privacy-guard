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
.venv/bin/pytest -q                                  # 123 tests
.venv/bin/python scripts/tamper_demo.py              # flip one audit line, watch verify() fail
```

## Results (seed 42 · 36 queries · answers qwen/qwen3.8-27b · judge openai/gpt-oss-120b)

Regenerate everything with `.venv/bin/python scripts/run_all.py` (uses Groq when a key or the committed cache is available).

| Configuration | Context PII exposure (to LLM provider) | Answer PII leakage (extractive) | Over-sharing retrieval rate | Injection exposure rate | Injection success rate (LLM) | Answer-quality retention (ROUGE-L) | Answer-quality retention (cosine) | Index PII exposure | Added latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla RAG | 100.0% | 33.2% | 50.9% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | +0.0 |
| Inherited RBAC (Copilot-style) | 49.2% | 19.6% | 0.6% | 59.4% | 12.5% | 69.2% | 76.8% | 100.0% | -0.4 |
| Redaction-only DLP | 0.0% | 0.0% | 50.9% | 100.0% | 12.5% | 50.1% | 60.6% | 100.0% | +39.6 |
| Output-side filtering | 100.0% | 0.0% | 50.9% | 100.0% | 0.0% | 87.6% | 87.8% | 100.0% | +7.7 |
| Delimiting only | 100.0% | 33.2% | 50.9% | 100.0% | 0.0% | 75.3% | 80.9% | 100.0% | -0.3 |
| Heuristic screening only | 86.0% | 36.7% | 53.8% | 0.0% | 0.0% | 70.9% | 78.6% | 100.0% | -0.2 |
| Copilot Privacy Guard (ours) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 34.9% | 47.8% | 0.0% | +97.4 |
| ours − LLM judge | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 34.9% | 47.8% | 0.0% | +97.7 |
| ours − access filter | 0.0% | 0.0% | 55.0% | 0.0% | 0.0% | 43.8% | 56.4% | 0.0% | +121.2 |
| ours − PII redaction | 48.8% | 19.7% | 0.0% | 0.0% | 0.0% | 49.7% | 63.2% | 100.0% | +58.3 |
| ours − redacted index | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 36.5% | 48.4% | 100.0% | +97.8 |
| ours on dense retrieval | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 34.4% | 48.3% | 0.0% | +130.0 |
| Configuration | Group | Analogue of |
|---|---|---|
| Vanilla RAG | comparison | Lewis et al. (2020) RAG; the undefended target of DEAL, ALDEN and Qi et al. |
| Inherited RBAC (Copilot-style) | comparison | Role-based access control (Sandhu et al., 1996) as inherited by M365 Copilot |
| Redaction-only DLP | comparison | Presidio-style / hybrid rule+NER PII redaction pipelines |
| Output-side filtering | comparison | Output guardrails; the generation-side goal of Privacy-Aware Decoding (simplified) |
| Delimiting only | comparison | Spotlighting (Hines et al., 2024) / StruQ-style separation without fine-tuning |
| Heuristic screening only | comparison | Pattern-based injection filters catalogued in prompt-injection surveys |
| Copilot Privacy Guard (ours) | ours | this work: the four Fig. 2 controls + privacy-preserving index |
| ours − LLM judge | ablation | ablation |
| ours − access filter | ablation | ablation |
| ours − PII redaction | ablation | ablation |
| ours − redacted index | ablation | ablation |
| ours on dense retrieval | ablation | retriever sensitivity (bge-small) |

PII detection (held-out F1): ours 89.5% · Presidio 54.3%. Injection detection: LLM judge 100% (FPR hard negatives 0%) · heuristic 33% · Prompt Guard 2 33%. Assurance checklist: 12/13 pass. Deliverables: `deliverables/`.

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
| `privacyguard/configs.py` | 12 comparative configurations (6 defence-family analogues, ours, 5 ablations) |
| `privacyguard/eval/injection_bench.py` · `pii_eval.py` · `heldout_pii.py` | Injection benchmark (heuristic vs LLM judge vs Prompt Guard 2); PII evaluation vs Presidio; held-out set |
| `privacyguard/compliance/` | Verified clause map, DPIA register, 13-check assurance checklist, evidence resolver |
| `scripts/run_all.py` · `make_figures.py` · `build_report.py` · `build_deck.py` | Regenerate results, figures, report and deck |
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
