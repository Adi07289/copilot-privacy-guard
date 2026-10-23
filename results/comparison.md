# Comparative evaluation

Every row runs the same 36 queries over the same corpus, retriever and LLM; only the defence differs. Rows other than *ours* are **simplified analogues of a defence family**, not reproductions of the cited papers.

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

Better: Context PII exposure (to LLM provider) → lower, Answer PII leakage (extractive) → lower, Over-sharing retrieval rate → lower, Injection exposure rate → lower, Injection success rate (LLM) → lower, Answer-quality retention (ROUGE-L) → higher, Answer-quality retention (cosine) → higher, Index PII exposure → lower, Added latency (ms) → lower

## What each row stands in for

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
