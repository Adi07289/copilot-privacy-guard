# DPIA risk register

Likelihood (L) and impact (I) are on a 1–5 scale and are **expert judgement**; residual scores cite the measured evidence that justifies them.

| ID | Threat | LINDDUN | OWASP | Inherent L×I | Controls | Residual L×I | Evidence | Rationale | Owner |
|---|---|---|---|---|---|---|---|---|---|
| R1 | Permission over-sharing | Disclosure of information; Non-compliance | LLM02, LLM06 | 4×4 = 16 (High) | access | 1×4 = 4 (Low) | none.oversharing_rate = 50.9%; full.oversharing_rate = 0.0% | Purpose-based filter removes every out-of-scope chunk on the evaluation set; tags are static, so impact stays high if a tag is wrong. | CISO / DPO |
| R2 | Prompt and context leakage (incl. to the LLM provider) | Disclosure of information | LLM02, LLM07 | 5×4 = 20 (Critical) | pii, index | 2×3 = 6 (Medium) | none.context_pii_exposure = 100.0%; full.context_pii_exposure = 0.0%; heldout.ours.recall = 85.7% | No raw PII reaches the provider on the corpus, but held-out recall is below 100%, so unfamiliar formats can still pass. | DPO |
| R3 | Indirect prompt injection | Detectability; Disclosure of information | LLM01 | 4×4 = 16 (High) | injection | 2×4 = 8 (Medium) | full.injection_success_rate = 0.0%; tpr.heuristic = 33.3%; tpr.combined = 100.0% | Seeded payloads are blocked and the judge adds coverage for paraphrased and Hindi payloads; novel attacks remain possible. | Security engineering |
| R4 | Embedding / index inversion | Identifiability; Disclosure of information | LLM08 | 3×4 = 12 (High) | index | 1×4 = 4 (Low) | none.index_pii_exposure = 100.0%; full.index_pii_exposure = 0.0% | The guarded index is built from redacted text, so inverting it cannot recover values that were never embedded. | Platform engineering |
| R5 | Fine-tuning memorisation | Identifiability; Disclosure of information | LLM02, LLM04 | 1×4 = 4 (Low) | none | 1×4 = 4 (Low) | — | No fine-tuning happens in this deployment; outside the control scope. | ML engineering |
| R6 | Sensitive-attribute inference by aggregation | Identifiability; Linkability | LLM02 | 3×4 = 12 (High) | access, pii | 2×3 = 6 (Medium) | full.oversharing_rate = 0.0% | Access control limits which fragments can be combined; aggregation inside an authorised scope is not controlled. | DPO |
| R7 | Audit-trail tampering | Non-compliance | — | 2×4 = 8 (Medium) | audit | 1×3 = 3 (Low) | A08 = PASS | Any edit or deletion breaks the hash chain and is reported; the log is tamper-evident, not tamper-proof. | Internal audit |

## Consistency check

- No residual score contradicts the evidence.
