"""Span-level PII-detector evaluation and a head-to-head against Microsoft Presidio (as shipped, and with its
India recognizers enabled). A prediction is a hit when it overlaps a gold span of the same (mapped) type."""
from __future__ import annotations
from collections import defaultdict
from typing import Callable
from privacyguard.controls.pii import PIIDetector
from privacyguard.eval.metrics import _overlaps
from privacyguard.models import Document, PIISpan

Detect = Callable[[str], list[PIISpan]]

PRESIDIO_TO_OURS = {"PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "IN_MOBILE",
                    "CREDIT_CARD": "CREDIT_CARD", "DATE_TIME": "DOB", "IN_AADHAAR": "AADHAAR", "IN_PAN": "PAN"}
OUR_TYPES = ("AADHAAR", "PAN", "IN_MOBILE", "IFSC", "UPI_VPA", "EMAIL", "CREDIT_CARD", "DOB", "SALARY_INR", "PERSON")


def _merge_same_type(spans: list[PIISpan]) -> list[PIISpan]:
    out: list[PIISpan] = []
    for s in sorted(spans, key=lambda s: (s.entity_type, s.start, -s.end)):
        if out and out[-1].entity_type == s.entity_type and s.start < out[-1].end:
            out[-1] = PIISpan(out[-1].start, max(out[-1].end, s.end), s.entity_type, out[-1].value)
        else:
            out.append(s)
    return out


def evaluate_detector(detect: Detect, docs: list[Document]) -> dict:
    tp = fp = fn = 0
    per: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    for d in docs:
        gold, matched = list(d.pii_spans), set()
        for p in _merge_same_type(detect(d.text)):
            hit = next((i for i, g in enumerate(gold) if i not in matched and _overlaps(p, g)), None)
            if hit is None:
                fp += 1; per[p.entity_type]["fp"] += 1
            else:
                tp += 1; matched.add(hit); per[gold[hit].entity_type]["tp"] += 1
        for i, g in enumerate(gold):
            if i not in matched:
                fn += 1; per[g.entity_type]["fn"] += 1
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
    per_type = {t: {"recall": round(v["tp"] / (v["tp"] + v["fn"]), 4) if v["tp"] + v["fn"] else None,
                    "support": v["tp"] + v["fn"], "fp": v["fp"]} for t, v in sorted(per.items())}
    return {"precision": round(p_, 4), "recall": round(r_, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
            "per_type": per_type}


def presidio_available() -> bool:
    try:
        import presidio_analyzer  # noqa: F401
        return True
    except Exception:
        return False


def presidio_detector(india: bool) -> Detect:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    nlp = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}).create_engine()
    engine = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    if india:
        from presidio_analyzer.predefined_recognizers import InAadhaarRecognizer, InPanRecognizer
        engine.registry.add_recognizer(InAadhaarRecognizer())
        engine.registry.add_recognizer(InPanRecognizer())

    def detect(text: str) -> list[PIISpan]:
        return [PIISpan(r.start, r.end, PRESIDIO_TO_OURS[r.entity_type], text[r.start:r.end])
                for r in engine.analyze(text=text, language="en") if r.entity_type in PRESIDIO_TO_OURS]
    return detect


def compare_detectors(datasets: dict[str, list[Document]]) -> dict:
    detectors: dict[str, Detect] = {"ours": PIIDetector().detect}
    version = None
    if presidio_available():
        import importlib.metadata as md
        version = md.version("presidio-analyzer")
        detectors["presidio_default"] = presidio_detector(False)
        detectors["presidio_india"] = presidio_detector(True)
    return {"detectors": list(detectors), "presidio_version": version, "mapping": PRESIDIO_TO_OURS,
            "types_without_presidio_recognizer": [t for t in OUR_TYPES if t not in PRESIDIO_TO_OURS.values()],
            "results": {ds: {n: evaluate_detector(fn, docs) for n, fn in detectors.items()}
                        for ds, docs in datasets.items()}}


_NAMES = {"ours": "Ours (India-first)", "presidio_default": "Presidio (as shipped)",
          "presidio_india": "Presidio + India recognizers"}


def to_markdown(res: dict) -> str:
    dets = res["detectors"]
    lines = ["# PII detection: ours vs Microsoft Presidio", "",
             f"Presidio version: {res['presidio_version'] or 'not installed'}. Presidio types mapped to ours: "
             + ", ".join(f"{k}→{v}" for k, v in res["mapping"].items())
             + ". No Presidio recognizer exists for: " + ", ".join(res["types_without_presidio_recognizer"]) + "."]
    for ds, by_det in res["results"].items():
        lines += ["", f"## Dataset: {ds}", "", "| Entity type | " + " | ".join(_NAMES[d] for d in dets) + " |",
                  "|---|" + "---|" * len(dets)]
        for t in OUR_TYPES:
            cells = []
            for d in dets:
                pt = by_det[d]["per_type"].get(t)
                cells.append("—" if not pt or pt["recall"] is None else f"{pt['recall']:.0%} ({pt['support']})")
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
        for m in ("precision", "recall", "f1"):
            lines.append(f"| **{m}** | " + " | ".join(f"{by_det[d][m]:.1%}" for d in dets) + " |")
    return "\n".join(lines) + "\n"
