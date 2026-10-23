# Compliance map

Paraphrases are ours, not legal text; each reference was verified (see *source*).

| Framework | Ref | Obligation | Controls | Evidence (measured) | Note |
|---|---|---|---|---|---|
| DPDP Act 2023 | §4 | **Grounds for processing.** Process personal data only for a lawful purpose, with consent or for a legitimate use. | Purpose-based access filter | full.oversharing_rate = 0.0% |  |
| DPDP Act 2023 | §6(1) | **Consent limited to necessary data.** Consent covers only the personal data necessary for the specified purpose. | PII detection & redaction (prompt + chunks), Purpose-based access filter | full.context_pii_exposure = 0.0%; full.oversharing_rate = 0.0% |  |
| DPDP Act 2023 | §7(i) | **Legitimate use: employment.** Employee data may be processed for employment purposes; the purpose still bounds the processing. | Purpose-based access filter | full.oversharing_rate = 0.0% | Basis for HR-purpose access to employee records. |
| DPDP Act 2023 | §8(1) | **Fiduciary responsible for its processors.** The Data Fiduciary remains responsible for processing carried out by its Data Processor (here, the LLM API). | PII detection & redaction (prompt + chunks), Privacy-preserving (redacted) index | full.context_pii_exposure = 0.0%; full.index_pii_exposure = 0.0% |  |
| DPDP Act 2023 | §8(5) | **Reasonable security safeguards.** Protect personal data in its possession or control, including at processors, with reasonable security safeguards to prevent a personal data breach. | PII detection & redaction (prompt + chunks), Purpose-based access filter, Injection screening (heuristic + LLM judge) | full.injection_success_rate = 0.0%; full.injection_exposure_rate = 0.0%; full.context_pii_exposure = 0.0% |  |
| DPDP Act 2023 | §8(6) | **Breach intimation.** Intimate the Board and each affected Data Principal of a personal data breach. | Hash-chained audit log | A08 = PASS | The audit chain supplies the forensic record; notification is a process. |
| DPDP Act 2023 | §10(2)(c) | **Significant Data Fiduciary: DPIA and audit.** A Significant Data Fiduciary undertakes periodic Data Protection Impact Assessments and audits. | Hash-chained audit log | results/dpia.md = results/dpia.md; A08 = PASS |  |
| DPDP Rules 2025 | Rule 6 | **Reasonable security safeguards.** Masking/obfuscation/tokenisation, access control, logging with monitoring and review, and log retention. | PII detection & redaction (prompt + chunks), Purpose-based access filter, Hash-chained audit log | full.context_pii_exposure = 0.0%; full.oversharing_rate = 0.0%; A08 = PASS; A10 = PASS |  |
| DPDP Rules 2025 | Rule 7 | **Intimation of personal data breach.** Notify the Board of a breach within the prescribed timelines with details of the breach. | Hash-chained audit log | A08 = PASS |  |
| DPDP Rules 2025 | Rule 13 | **Additional obligations of Significant Data Fiduciaries.** Annual DPIA and audit; due diligence that algorithmic software does not risk Data Principals' rights. | Hash-chained audit log | results/dpia.md = results/dpia.md; results/checklist.md = results/checklist.md |  |
| GDPR | Art. 5(1)(b) | **Purpose limitation.** Collected for specified purposes; no incompatible further processing. | Purpose-based access filter | full.oversharing_rate = 0.0% |  |
| GDPR | Art. 5(1)(c) | **Data minimisation.** Adequate, relevant and limited to what is necessary. | PII detection & redaction (prompt + chunks), Privacy-preserving (redacted) index | full.context_pii_exposure = 0.0%; full.index_pii_exposure = 0.0% |  |
| GDPR | Art. 5(1)(f) | **Integrity and confidentiality.** Appropriate security against unauthorised processing. | PII detection & redaction (prompt + chunks), Purpose-based access filter, Injection screening (heuristic + LLM judge) | full.injection_success_rate = 0.0%; full.oversharing_rate = 0.0% |  |
| GDPR | Art. 5(2) | **Accountability.** The controller must be able to demonstrate compliance. | Hash-chained audit log | A08 = PASS; A10 = PASS |  |
| GDPR | Art. 25 | **Data protection by design and by default.** Technical measures such as pseudonymisation built into processing. | PII detection & redaction (prompt + chunks), Privacy-preserving (redacted) index, Purpose-based access filter | full.context_pii_exposure = 0.0%; full.index_pii_exposure = 0.0% |  |
| GDPR | Art. 30 | **Records of processing activities.** Maintain a record of processing activities. | Hash-chained audit log | A10 = PASS |  |
| GDPR | Art. 32 | **Security of processing.** Pseudonymisation, confidentiality, and regular testing of the effectiveness of measures. | PII detection & redaction (prompt + chunks), Injection screening (heuristic + LLM judge) | heldout.ours.recall = 85.7%; results/checklist.md = results/checklist.md |  |
| GDPR | Art. 35 | **Data protection impact assessment.** DPIA for processing likely to be high-risk. | Hash-chained audit log | results/dpia.md = results/dpia.md |  |
| EU AI Act | Art. 12 | **Record-keeping.** High-risk AI systems allow automatic recording of events (logs). | Hash-chained audit log | A08 = PASS | Applies if the deployment is high-risk (see Annex III(4)). |
| EU AI Act | Art. 15 | **Accuracy, robustness and cybersecurity.** Resilience against attempts by third parties to alter use or outputs by exploiting vulnerabilities. | Injection screening (heuristic + LLM judge) | full.injection_success_rate = 0.0%; tpr.combined = 100.0% |  |
| EU AI Act | Annex III(4) | **Employment and workers management.** AI used for decisions affecting work relationships or to monitor workers is classified high-risk. | Purpose-based access filter, Hash-chained audit log | full.oversharing_rate = 0.0% | A copilot used for HR decisions could fall here; answering questions alone may not. |
| OWASP LLM Top 10 (2025) | LLM01 | **Prompt injection.** Direct and indirect injection via retrieved content. | Injection screening (heuristic + LLM judge) | full.injection_success_rate = 0.0%; tpr.combined = 100.0%; fpr_all.combined = 10.4% |  |
| OWASP LLM Top 10 (2025) | LLM02 | **Sensitive information disclosure.** PII or confidential data in outputs. | PII detection & redaction (prompt + chunks), Purpose-based access filter | full.pii_leakage_rate = 0.0%; full.context_pii_exposure = 0.0% |  |
| OWASP LLM Top 10 (2025) | LLM08 | **Vector and embedding weaknesses.** Leakage and inversion via the vector store; cross-tenant/permission issues in RAG. | Privacy-preserving (redacted) index, Purpose-based access filter | full.index_pii_exposure = 0.0% |  |

## Sources

- DPDP Act 2023 §4: https://www.dpdpa.com/dpdpa2023/chapter-2/section4.html
- DPDP Act 2023 §6(1): https://www.dpdpa.com/dpdpa2023/chapter-2/section6.html
- DPDP Act 2023 §7(i): https://www.dpdpa.com/dpdpa2023/chapter-2/section7.html
- DPDP Act 2023 §8(1): https://www.dpdpa.com/dpdpa2023/chapter-2/section8.html
- DPDP Act 2023 §8(5): https://www.dpdpa.com/dpdpa2023/chapter-2/section8.html
- DPDP Act 2023 §8(6): https://www.dpdpa.com/dpdpa2023/chapter-2/section8.html
- DPDP Act 2023 §10(2)(c): https://www.dpdpa.com/dpdpa2023/chapter-2/section10.html
- DPDP Rules 2025 Rule 6: https://www.dpdpa.com/dpdparules/rule6.html
- DPDP Rules 2025 Rule 7: https://www.dpdpa.com/dpdparules/rule7.html
- DPDP Rules 2025 Rule 13: https://www.dpdpa.com/dpdparules/rule13.html
- GDPR Art. 5(1)(b): https://gdpr-info.eu/art-5-gdpr/
- GDPR Art. 5(1)(c): https://gdpr-info.eu/art-5-gdpr/
- GDPR Art. 5(1)(f): https://gdpr-info.eu/art-5-gdpr/
- GDPR Art. 5(2): https://gdpr-info.eu/art-5-gdpr/
- GDPR Art. 25: https://gdpr-info.eu/art-25-gdpr/
- GDPR Art. 30: https://gdpr-info.eu/art-30-gdpr/
- GDPR Art. 32: https://gdpr-info.eu/art-32-gdpr/
- GDPR Art. 35: https://gdpr-info.eu/art-35-gdpr/
- EU AI Act Art. 12: https://artificialintelligenceact.eu/article/12/
- EU AI Act Art. 15: https://artificialintelligenceact.eu/article/15/
- EU AI Act Annex III(4): https://artificialintelligenceact.eu/annex/3/
- OWASP LLM Top 10 (2025) LLM01: https://genai.owasp.org/llmrisk/llm01-prompt-injection/
- OWASP LLM Top 10 (2025) LLM02: https://genai.owasp.org/llmrisk/llm022025-sensitive-information-disclosure/
- OWASP LLM Top 10 (2025) LLM08: https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/
