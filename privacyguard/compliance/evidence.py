"""Resolves evidence keys against the result files so every compliance claim cites a measured value."""
from __future__ import annotations
from dataclasses import dataclass, field
from privacyguard.llm import ROOT

_PERCENT_EXEMPT = {"added_latency_ms", "injection_success_n", "injection_success_excluded_fallback", "n_runs"}


@dataclass
class EvidenceStore:
    comparison: dict                      # merged["aggregates"]: config -> metric -> value
    bench: dict | None = None
    pii: dict | None = None               # compare_detectors() output
    checks: dict = field(default_factory=dict)   # check id -> CheckResult

    def value(self, key: str):
        kind, _, path = key.partition(":")
        if kind == "check":
            return self.checks[path].passed
        if kind == "artifact":
            if not (ROOT / path).exists():
                raise KeyError(key)
            return path
        roots = {"metric": self.comparison, "bench": self.bench, "pii": (self.pii or {}).get("results")}
        node = roots[kind]
        if node is None:
            raise KeyError(key)
        for part in path.split("."):
            node = node[part]
        return node

    def fmt(self, key: str) -> str:
        v = self.value(key)
        if key.startswith("check:"):
            return {True: "PASS", False: "FAIL", None: "not measured"}[v]
        if v is None:
            return "not measured"
        if isinstance(v, float) and key.rsplit(".", 1)[-1] not in _PERCENT_EXEMPT:
            return f"{v:.1%}"
        return str(v)
