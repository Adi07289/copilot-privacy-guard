# Roadmap — status at final review

| # | Objective | Status | Artifact |
|---|---|---|---|
| O1 | Literature review | Done — 32 verified papers | report §Literature review, `docs/report/references.py` |
| O2 | Data-flow map | Done | Review 2 Fig. 1; report Fig. 1 |
| O3 | Threat taxonomy (LINDDUN + OWASP) | Done | Review 2 §6.2; `privacyguard/compliance/dpia.py` |
| O4 | Incident analyses | Done — 3 incidents | `docs/INCIDENTS.md` |
| O5 | Prototype middleware | Done — 4 controls + redacted index, 123 tests | `privacyguard/` |
| O6 | Evaluation protocol applied | Done — 9 metrics × 12 configs | `results/comparison.md` |
| O7 | DPIA risk register | Done — 7 risks | `results/dpia.md` |
| O8 | Clause mapping | Done — 24 verified clauses | `results/compliance.md` |
| O9 | Privacy Assurance Checklist | Done — 12/13 pass | `results/checklist.md` |
| O10 | Limitations & future work | Done | report §Future work, deck slides 16–17 |

## Genuine future work
1. Judge as sole drop decision (removes heuristic false positives). 2. Re-identify placeholders for authorised users.
3. Recognizers for secrets, spaced mobiles, written dates, lakh/crore. 4. Output egress allow-list.
5. Identity-provider ACLs + purpose consent prompt. 6. Real de-identified corpus; adaptive attacks. 7. Batch/cache judge calls.
