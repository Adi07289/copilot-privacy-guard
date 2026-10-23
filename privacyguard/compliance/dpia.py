"""DPIA-style risk register. Likelihood and impact are expert judgement (1-5); residual scores cite the measured
metric that justifies them, and consistency_warnings() flags a residual that the evidence does not support."""
from __future__ import annotations
from dataclasses import dataclass

LOWER_IS_BETTER = {"context_pii_exposure", "pii_leakage_rate", "oversharing_rate", "injection_exposure_rate",
                   "injection_success_rate", "index_pii_exposure"}


def level(score: int) -> str:
    return "Low" if score <= 4 else "Medium" if score <= 9 else "High" if score <= 16 else "Critical"


@dataclass(frozen=True)
class Risk:
    id: str
    threat: str
    linddun: str
    owasp: str
    likelihood: int
    impact: int
    controls: tuple[str, ...]
    residual_likelihood: int
    residual_impact: int
    evidence: tuple[str, ...]
    rationale: str
    owner: str

    @property
    def inherent(self) -> int:
        return self.likelihood * self.impact

    @property
    def residual(self) -> int:
        return self.residual_likelihood * self.residual_impact


RISKS: list[Risk] = [
    Risk("R1", "Permission over-sharing", "Disclosure of information; Non-compliance", "LLM02, LLM06", 4, 4,
         ("access",), 1, 4, ("metric:none.oversharing_rate", "metric:full.oversharing_rate"),
         "Purpose-based filter removes every out-of-scope chunk on the evaluation set; tags are static, so impact "
         "stays high if a tag is wrong.", "CISO / DPO"),
    Risk("R2", "Prompt and context leakage (incl. to the LLM provider)", "Disclosure of information", "LLM02, LLM07",
         5, 4, ("pii", "index"), 2, 3,
         ("metric:none.context_pii_exposure", "metric:full.context_pii_exposure", "pii:heldout.ours.recall"),
         "No raw PII reaches the provider on the corpus, but held-out recall is below 100%, so unfamiliar formats "
         "can still pass.", "DPO"),
    Risk("R3", "Indirect prompt injection", "Detectability; Disclosure of information", "LLM01", 4, 4,
         ("injection",), 2, 4,
         ("metric:full.injection_success_rate", "bench:tpr.heuristic", "bench:tpr.combined"),
         "Seeded payloads are blocked and the judge adds coverage for paraphrased and Hindi payloads; novel attacks "
         "remain possible.", "Security engineering"),
    Risk("R4", "Embedding / index inversion", "Identifiability; Disclosure of information", "LLM08", 3, 4,
         ("index",), 1, 4, ("metric:none.index_pii_exposure", "metric:full.index_pii_exposure"),
         "The guarded index is built from redacted text, so inverting it cannot recover values that were never "
         "embedded.", "Platform engineering"),
    Risk("R5", "Fine-tuning memorisation", "Identifiability; Disclosure of information", "LLM02, LLM04", 1, 4,
         (), 1, 4, (), "No fine-tuning happens in this deployment; outside the control scope.", "ML engineering"),
    Risk("R6", "Sensitive-attribute inference by aggregation", "Identifiability; Linkability", "LLM02", 3, 4,
         ("access", "pii"), 2, 3, ("metric:full.oversharing_rate",),
         "Access control limits which fragments can be combined; aggregation inside an authorised scope is not "
         "controlled.", "DPO"),
    Risk("R7", "Audit-trail tampering", "Non-compliance", "—", 2, 4, ("audit",), 1, 3, ("check:A08",),
         "Any edit or deletion breaks the hash chain and is reported; the log is tamper-evident, not tamper-proof.",
         "Internal audit"),
]


def consistency_warnings(risks: list[Risk], store) -> list[str]:
    out = []
    for r in risks:
        if r.residual_likelihood > 1:
            continue
        for key in r.evidence:
            if key.startswith("metric:full."):
                metric = key.rsplit(".", 1)[-1]
                v = store.value(key)
                if metric in LOWER_IS_BETTER and isinstance(v, (int, float)) and v > 0:
                    out.append(f"{r.id}: residual likelihood 1 but {key} = {store.fmt(key)}")
    return out
