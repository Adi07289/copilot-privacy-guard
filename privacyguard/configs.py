"""Comparative configurations. Every non-`full` row is a simplified analogue of a defence family from the
literature, run on the same corpus, queries, retriever and LLM - not a reproduction of the cited paper."""
from __future__ import annotations
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GuardConfig:
    name: str
    label: str
    analogue_of: str
    redact_prompt: bool = False
    redact_index: bool = False
    access: str = "none"              # none | rbac | purpose
    injection: str = "none"           # none | spotlight | heuristic | heuristic+judge
    redact_chunks: bool = False
    output_filter: bool = False
    audit: bool = False
    retriever: str = "tfidf"          # tfidf | dense
    group: str = "comparison"         # comparison | ours | ablation


_FULL = GuardConfig(
    "full", "Copilot Privacy Guard (ours)", "this work: the four Fig. 2 controls + privacy-preserving index",
    redact_prompt=True, redact_index=True, access="purpose", injection="heuristic+judge",
    redact_chunks=True, audit=True, group="ours")

_ALL = [
    GuardConfig("none", "Vanilla RAG", "Lewis et al. (2020) RAG; the undefended target of DEAL, ALDEN and Qi et al."),
    GuardConfig("rbac", "Inherited RBAC (Copilot-style)",
                "Role-based access control (Sandhu et al., 1996) as inherited by M365 Copilot", access="rbac"),
    GuardConfig("pii_only", "Redaction-only DLP", "Presidio-style / hybrid rule+NER PII redaction pipelines",
                redact_prompt=True, redact_chunks=True),
    GuardConfig("output_filter", "Output-side filtering",
                "Output guardrails; the generation-side goal of Privacy-Aware Decoding (simplified)", output_filter=True),
    GuardConfig("spotlight", "Delimiting only",
                "Spotlighting (Hines et al., 2024) / StruQ-style separation without fine-tuning", injection="spotlight"),
    GuardConfig("heuristic", "Heuristic screening only",
                "Pattern-based injection filters catalogued in prompt-injection surveys", injection="heuristic"),
    _FULL,
    replace(_FULL, name="full_no_judge", label="ours − LLM judge", analogue_of="ablation",
            injection="heuristic", group="ablation"),
    replace(_FULL, name="full_no_access", label="ours − access filter", analogue_of="ablation",
            access="none", group="ablation"),
    replace(_FULL, name="full_no_pii", label="ours − PII redaction", analogue_of="ablation",
            redact_prompt=False, redact_index=False, redact_chunks=False, group="ablation"),
    replace(_FULL, name="full_no_index", label="ours − redacted index", analogue_of="ablation",
            redact_index=False, group="ablation"),
    replace(_FULL, name="full_dense", label="ours on dense retrieval", analogue_of="retriever sensitivity (bge-small)",
            retriever="dense", group="ablation"),
]

CONFIGS: dict[str, GuardConfig] = {c.name: c for c in _ALL}
CONFIG_ORDER: list[str] = [c.name for c in _ALL]
