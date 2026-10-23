"""Clause-level compliance map: obligation -> control -> measured evidence. References verified by web search;
paraphrases are ours, not legal text."""
from __future__ import annotations
from dataclasses import dataclass

CONTROL_IDS = ("pii", "index", "access", "injection", "audit")
CONTROL_NAMES = {"pii": "PII detection & redaction (prompt + chunks)", "index": "Privacy-preserving (redacted) index",
                 "access": "Purpose-based access filter", "injection": "Injection screening (heuristic + LLM judge)",
                 "audit": "Hash-chained audit log"}


@dataclass(frozen=True)
class Clause:
    framework: str
    ref: str
    title: str
    obligation: str
    controls: tuple[str, ...]
    evidence: tuple[str, ...]
    note: str = ""
    source: str = ""


CLAUSES: list[Clause] = [
    Clause("DPDP Act 2023", "§4", "Grounds for processing",
           "Process personal data only for a lawful purpose, with consent or for a legitimate use.",
           ("access",), ("metric:full.oversharing_rate",),
           source="https://www.dpdpa.com/dpdpa2023/chapter-2/section4.html"),
    Clause("DPDP Act 2023", "§6(1)", "Consent limited to necessary data",
           "Consent covers only the personal data necessary for the specified purpose.",
           ("pii", "access"), ("metric:full.context_pii_exposure", "metric:full.oversharing_rate"),
           source="https://www.dpdpa.com/dpdpa2023/chapter-2/section6.html"),
    Clause("DPDP Act 2023", "§7(i)", "Legitimate use: employment",
           "Employee data may be processed for employment purposes; the purpose still bounds the processing.",
           ("access",), ("metric:full.oversharing_rate",), note="Basis for HR-purpose access to employee records.",
           source="https://www.dpdpa.com/dpdpa2023/chapter-2/section7.html"),
    Clause("DPDP Act 2023", "§8(1)", "Fiduciary responsible for its processors",
           "The Data Fiduciary remains responsible for processing carried out by its Data Processor (here, the LLM API).",
           ("pii", "index"), ("metric:full.context_pii_exposure", "metric:full.index_pii_exposure"),
           source="https://www.dpdpa.com/dpdpa2023/chapter-2/section8.html"),
    Clause("DPDP Act 2023", "§8(5)", "Reasonable security safeguards",
           "Protect personal data in its possession or control, including at processors, with reasonable security "
           "safeguards to prevent a personal data breach.",
           ("pii", "access", "injection"),
           ("metric:full.injection_success_rate", "metric:full.injection_exposure_rate", "metric:full.context_pii_exposure"),
           source="https://www.dpdpa.com/dpdpa2023/chapter-2/section8.html"),
    Clause("DPDP Act 2023", "§8(6)", "Breach intimation",
           "Intimate the Board and each affected Data Principal of a personal data breach.",
           ("audit",), ("check:A08",), note="The audit chain supplies the forensic record; notification is a process.",
           source="https://www.dpdpa.com/dpdpa2023/chapter-2/section8.html"),
    Clause("DPDP Act 2023", "§10(2)(c)", "Significant Data Fiduciary: DPIA and audit",
           "A Significant Data Fiduciary undertakes periodic Data Protection Impact Assessments and audits.",
           ("audit",), ("artifact:results/dpia.md", "check:A08"),
           source="https://www.dpdpa.com/dpdpa2023/chapter-2/section10.html"),
    Clause("DPDP Rules 2025", "Rule 6", "Reasonable security safeguards",
           "Masking/obfuscation/tokenisation, access control, logging with monitoring and review, and log retention.",
           ("pii", "access", "audit"),
           ("metric:full.context_pii_exposure", "metric:full.oversharing_rate", "check:A08", "check:A10"),
           source="https://www.dpdpa.com/dpdparules/rule6.html"),
    Clause("DPDP Rules 2025", "Rule 7", "Intimation of personal data breach",
           "Notify the Board of a breach within the prescribed timelines with details of the breach.",
           ("audit",), ("check:A08",),
           source="https://www.dpdpa.com/dpdparules/rule7.html"),
    Clause("DPDP Rules 2025", "Rule 13", "Additional obligations of Significant Data Fiduciaries",
           "Annual DPIA and audit; due diligence that algorithmic software does not risk Data Principals' rights.",
           ("audit",), ("artifact:results/dpia.md", "artifact:results/checklist.md"),
           source="https://www.dpdpa.com/dpdparules/rule13.html"),
    Clause("GDPR", "Art. 5(1)(b)", "Purpose limitation", "Collected for specified purposes; no incompatible further processing.",
           ("access",), ("metric:full.oversharing_rate",),
           source="https://gdpr-info.eu/art-5-gdpr/"),
    Clause("GDPR", "Art. 5(1)(c)", "Data minimisation", "Adequate, relevant and limited to what is necessary.",
           ("pii", "index"), ("metric:full.context_pii_exposure", "metric:full.index_pii_exposure"),
           source="https://gdpr-info.eu/art-5-gdpr/"),
    Clause("GDPR", "Art. 5(1)(f)", "Integrity and confidentiality", "Appropriate security against unauthorised processing.",
           ("pii", "access", "injection"), ("metric:full.injection_success_rate", "metric:full.oversharing_rate"),
           source="https://gdpr-info.eu/art-5-gdpr/"),
    Clause("GDPR", "Art. 5(2)", "Accountability", "The controller must be able to demonstrate compliance.",
           ("audit",), ("check:A08", "check:A10"),
           source="https://gdpr-info.eu/art-5-gdpr/"),
    Clause("GDPR", "Art. 25", "Data protection by design and by default",
           "Technical measures such as pseudonymisation built into processing.", ("pii", "index", "access"),
           ("metric:full.context_pii_exposure", "metric:full.index_pii_exposure"),
           source="https://gdpr-info.eu/art-25-gdpr/"),
    Clause("GDPR", "Art. 30", "Records of processing activities", "Maintain a record of processing activities.",
           ("audit",), ("check:A10",),
           source="https://gdpr-info.eu/art-30-gdpr/"),
    Clause("GDPR", "Art. 32", "Security of processing",
           "Pseudonymisation, confidentiality, and regular testing of the effectiveness of measures.",
           ("pii", "injection"), ("pii:heldout.ours.recall", "artifact:results/checklist.md"),
           source="https://gdpr-info.eu/art-32-gdpr/"),
    Clause("GDPR", "Art. 35", "Data protection impact assessment", "DPIA for processing likely to be high-risk.",
           ("audit",), ("artifact:results/dpia.md",),
           source="https://gdpr-info.eu/art-35-gdpr/"),
    Clause("EU AI Act", "Art. 12", "Record-keeping", "High-risk AI systems allow automatic recording of events (logs).",
           ("audit",), ("check:A08",), note="Applies if the deployment is high-risk (see Annex III(4)).",
           source="https://artificialintelligenceact.eu/article/12/"),
    Clause("EU AI Act", "Art. 15", "Accuracy, robustness and cybersecurity",
           "Resilience against attempts by third parties to alter use or outputs by exploiting vulnerabilities.",
           ("injection",), ("metric:full.injection_success_rate", "bench:tpr.combined"),
           source="https://artificialintelligenceact.eu/article/15/"),
    Clause("EU AI Act", "Annex III(4)", "Employment and workers management",
           "AI used for decisions affecting work relationships or to monitor workers is classified high-risk.",
           ("access", "audit"), ("metric:full.oversharing_rate",),
           note="A copilot used for HR decisions could fall here; answering questions alone may not.",
           source="https://artificialintelligenceact.eu/annex/3/"),
    Clause("OWASP LLM Top 10 (2025)", "LLM01", "Prompt injection", "Direct and indirect injection via retrieved content.",
           ("injection",), ("metric:full.injection_success_rate", "bench:tpr.combined", "bench:fpr_all.combined"),
           source="https://genai.owasp.org/llmrisk/llm01-prompt-injection/"),
    Clause("OWASP LLM Top 10 (2025)", "LLM02", "Sensitive information disclosure", "PII or confidential data in outputs.",
           ("pii", "access"), ("metric:full.pii_leakage_rate", "metric:full.context_pii_exposure"),
           source="https://genai.owasp.org/llmrisk/llm022025-sensitive-information-disclosure/"),
    Clause("OWASP LLM Top 10 (2025)", "LLM08", "Vector and embedding weaknesses",
           "Leakage and inversion via the vector store; cross-tenant/permission issues in RAG.",
           ("index", "access"), ("metric:full.index_pii_exposure",),
           source="https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/"),
]
