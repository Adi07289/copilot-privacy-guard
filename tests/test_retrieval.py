from privacyguard.models import Document
from privacyguard.retrieval import TfidfIndex

def _doc(i, text, dept="Sales"):
    return Document(id=f"d{i}", kind="email", department=dept, sensitivity=1, purposes=["general"], title="t", text=text)

def test_search_ranks_relevant_first_and_copies_tags():
    idx = TfidfIndex()
    idx.index([_doc(1, "payroll salary revision april"), _doc(2, "kubernetes deployment pipeline"), _doc(3, "salary slip and ctc")])
    res = idx.search("salary", k=2)
    assert [c.doc_id for c in res] == ["d3", "d1"] or [c.doc_id for c in res] == ["d1", "d3"]
    assert res[0].department == "Sales" and res[0].score > 0

def test_search_before_index_is_empty():
    assert TfidfIndex().search("x") == []
