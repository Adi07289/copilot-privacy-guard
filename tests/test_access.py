from privacyguard.controls.access import AccessPolicy, PurposeBasedAccessFilter
from privacyguard.models import Chunk, GuardContext

def ch(dept="Engineering", sens=1, purposes=("engineering",)):
    return Chunk("d", "t", 1.0, dept, sens, list(purposes))

def test_reasons_in_order(engineer):
    p = AccessPolicy()
    assert p.allows(engineer, "engineering", ch()) == (True, None)
    assert p.allows(engineer, "engineering", ch(sens=3)) == (False, "clearance")
    assert p.allows(engineer, "engineering", ch(dept="HR", sens=1, purposes=("hr_operations",))) == (False, "department")
    assert p.allows(engineer, "engineering", ch(purposes=("finance_reporting",))) == (False, "purpose")

def test_public_and_general_and_cross_dept(engineer, ciso):
    p = AccessPolicy()
    assert p.allows(engineer, "engineering", ch(dept="Sales", sens=0, purposes=("general",)))[0]
    assert p.allows(engineer, "sales_support", ch(purposes=("engineering", "general")))[0]
    assert p.allows(ciso, "legal_review", ch(dept="HR", sens=3, purposes=("legal_review",)))[0]
    assert p.allows(ciso, "engineering", ch(dept="HR", sens=3, purposes=("hr_operations",))) == (False, "purpose")

def test_filter_marks_and_records(engineer):
    ctx = GuardContext(user=engineer, purpose="engineering", query="q",
                       chunks=[ch(), ch(dept="HR", sens=3, purposes=("hr_operations",))])
    ctx = PurposeBasedAccessFilter().apply(ctx)
    ok, bad = ctx.chunks
    assert not ok.dropped and bad.dropped and bad.drop_reason == "clearance" and "out_of_scope" in bad.flags
    acts = [(d.action, d.detail.get("reason")) for d in ctx.decisions]
    assert acts == [("allow", None), ("drop", "clearance")]
