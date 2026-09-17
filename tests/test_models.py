from privacyguard.models import Document, PIISpan, User, Chunk, GuardContext, Decision, CROSS_DEPT_ROLES

def test_document_roundtrip():
    d = Document(id="hr-001", kind="hr", department="HR", sensitivity=3,
                 purposes=["hr_operations"], title="Payroll", text="Salary of X is 5",
                 pii_spans=[PIISpan(0, 6, "SALARY_INR", "Salary")], is_poisoned=False)
    assert Document.from_dict(d.to_dict()) == d

def test_chunk_from_document_copies_tags():
    d = Document(id="e1", kind="email", department="Sales", sensitivity=1, purposes=["general"],
                 title="t", text="hello", pii_spans=[])
    c = Chunk.from_document(d, score=0.5)
    assert (c.doc_id, c.department, c.sensitivity, c.purposes, c.dropped) == ("e1", "Sales", 1, ["general"], False)

def test_guard_context_defaults():
    u = User(id="u1", name="A", role="Engineer", department="Engineering", clearance=1)
    ctx = GuardContext(user=u, purpose="engineering", query="q")
    assert ctx.chunks == [] and ctx.decisions == [] and ctx.redacted_query is None
    assert "CISO" in CROSS_DEPT_ROLES
