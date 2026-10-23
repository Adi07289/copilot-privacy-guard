# Injection-detection benchmark

24 payloads (6 families × 4) in a benign ticket carrier; 48 negatives (clean corpus documents + 8 hard negatives). Threshold 0.5. LLM judge available: True.

| Family | Heuristic | LLM judge (gpt-oss-120b) | Combined (ours) | Prompt Guard 2 (Meta) |
|---|---|---|---|---|
| direct | 50% | 100% | 100% | 75% |
| roleplay | 50% | 100% | 100% | 50% |
| exfiltration | 25% | 100% | 100% | 0% |
| obfuscated | 75% | 100% | 100% | 25% |
| paraphrased | 0% | 100% | 100% | 50% |
| hindi | 0% | 100% | 100% | 0% |
| **Detection rate (all payloads)** | 33% | 100% | 100% | 33% |
| False-positive rate (all negatives) | 10% | 0% | 10% | 2% |
| False-positive rate (clean docs) | 0% | 0% | 0% | 0% |
| False-positive rate (hard negatives) | 62% | 0% | 62% | 12% |
