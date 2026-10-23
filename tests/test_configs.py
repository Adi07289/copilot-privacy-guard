from dataclasses import fields
from privacyguard.configs import CONFIGS, CONFIG_ORDER

def _diff(a, b):
    return {f.name for f in fields(a) if getattr(a, f.name) != getattr(b, f.name)} - {"name", "label", "analogue_of", "group"}

def test_registry_shape():
    assert CONFIG_ORDER[0] == "none" and len(CONFIG_ORDER) == len(set(CONFIG_ORDER)) == 12
    f = CONFIGS["full"]
    assert (f.redact_prompt, f.redact_index, f.access, f.injection, f.redact_chunks, f.audit) == \
           (True, True, "purpose", "heuristic+judge", True, True)
    assert all(c.analogue_of for c in CONFIGS.values())

def test_ablations_change_only_their_control():
    f = CONFIGS["full"]
    assert _diff(CONFIGS["full_no_judge"], f) == {"injection"}
    assert _diff(CONFIGS["full_no_access"], f) == {"access"}
    assert _diff(CONFIGS["full_no_pii"], f) == {"redact_prompt", "redact_index", "redact_chunks"}
    assert _diff(CONFIGS["full_no_index"], f) == {"redact_index"}
    assert _diff(CONFIGS["full_dense"], f) == {"retriever"}
