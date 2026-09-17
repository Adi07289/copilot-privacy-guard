"""Core dataclasses shared by corpus, controls, pipeline and eval."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal

SENSITIVITY_LABELS = {0: "public", 1: "internal", 2: "confidential", 3: "restricted"}
DEPARTMENTS = ["HR", "Engineering", "Finance", "Legal", "Sales"]
PURPOSES = ["hr_operations", "engineering", "finance_reporting", "legal_review", "sales_support", "general"]
CROSS_DEPT_ROLES = {"CISO", "DPO"}
DocKind = Literal["email", "hr", "ticket", "code"]


@dataclass
class PIISpan:
    start: int
    end: int
    entity_type: str
    value: str


@dataclass
class Document:
    id: str
    kind: str
    department: str
    sensitivity: int
    purposes: list[str]
    title: str
    text: str
    pii_spans: list[PIISpan] = field(default_factory=list)
    is_poisoned: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        d = dict(d)
        d["pii_spans"] = [PIISpan(**s) for s in d.get("pii_spans", [])]
        return cls(**d)


@dataclass
class User:
    id: str
    name: str
    role: str
    department: str
    clearance: int


@dataclass
class Chunk:
    doc_id: str
    text: str
    score: float
    department: str
    sensitivity: int
    purposes: list[str]
    is_poisoned: bool = False
    flags: list[str] = field(default_factory=list)
    dropped: bool = False
    drop_reason: str | None = None

    @classmethod
    def from_document(cls, doc: Document, score: float) -> "Chunk":
        return cls(doc_id=doc.id, text=doc.text, score=score, department=doc.department,
                   sensitivity=doc.sensitivity, purposes=list(doc.purposes), is_poisoned=doc.is_poisoned)


@dataclass
class Decision:
    control: str
    action: str
    doc_id: str | None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GuardContext:
    user: User
    purpose: str
    query: str
    redacted_query: str | None = None
    chunks: list[Chunk] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    pii_map: dict[str, str] = field(default_factory=dict)
