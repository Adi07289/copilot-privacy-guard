from privacyguard.controls.access import AccessPolicy
from privacyguard.controls.pii import PIIRedactor
from privacyguard.corpus.generator import generate_corpus, get_persona
from privacyguard.eval.queries import QUERY_SET
from privacyguard.retrieval import build_indexes

def test_injection_queries_reach_authorized_poisoned_docs():
    corpus = generate_corpus(42, 80)
    idx = build_indexes(corpus, PIIRedactor())
    policy, hits = AccessPolicy("purpose"), 0
    for q in [q for q in QUERY_SET if q.kind == "injection"]:
        user = get_persona(q.user_id)
        for key in ("tfidf:raw", "tfidf:redacted"):
            chunks = idx[key].search(q.text, 6)
            if any(c.is_poisoned and policy.allows(user, q.purpose, c)[0] for c in chunks):
                hits += 1
    assert hits == 16, hits      # all 8 injection queries, on both indexes

def test_unauthorized_probes_are_unauthorized_for_hr_docs():
    for q in [q for q in QUERY_SET if q.kind == "unauthorized_probe"]:
        assert get_persona(q.user_id).department != "HR"
