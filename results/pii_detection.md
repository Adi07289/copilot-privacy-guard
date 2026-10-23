# PII detection: ours vs Microsoft Presidio

Presidio version: 2.2.364. Presidio types mapped to ours: PERSON→PERSON, EMAIL_ADDRESS→EMAIL, PHONE_NUMBER→IN_MOBILE, CREDIT_CARD→CREDIT_CARD, DATE_TIME→DOB, IN_AADHAAR→AADHAAR, IN_PAN→PAN. No Presidio recognizer exists for: IFSC, UPI_VPA, SALARY_INR.

## Dataset: corpus

| Entity type | Ours (India-first) | Presidio (as shipped) | Presidio + India recognizers |
|---|---|---|---|
| AADHAAR | 100% (20) | 0% (20) | 100% (20) |
| PAN | 100% (31) | 0% (31) | 100% (31) |
| IN_MOBILE | 100% (80) | 100% (80) | 100% (80) |
| IFSC | 95% (20) | 0% (20) | 0% (20) |
| UPI_VPA | 90% (20) | 0% (20) | 0% (20) |
| EMAIL | 100% (60) | 100% (60) | 100% (60) |
| CREDIT_CARD | 100% (8) | 100% (8) | 100% (8) |
| DOB | 95% (20) | 100% (20) | 100% (20) |
| SALARY_INR | 100% (20) | 0% (20) | 0% (20) |
| PERSON | 100% (90) | 67% (90) | 67% (90) |
| **precision** | 92.9% | 65.0% | 53.4% |
| **recall** | 98.9% | 61.8% | 75.6% |
| **f1** | 95.8% | 63.3% | 62.6% |

## Dataset: heldout

| Entity type | Ours (India-first) | Presidio (as shipped) | Presidio + India recognizers |
|---|---|---|---|
| AADHAAR | 100% (3) | 0% (3) | 100% (3) |
| PAN | 67% (3) | 0% (3) | 100% (3) |
| IN_MOBILE | 80% (5) | 100% (5) | 100% (5) |
| IFSC | 100% (2) | 0% (2) | 0% (2) |
| UPI_VPA | 100% (3) | 0% (3) | 0% (3) |
| EMAIL | 100% (2) | 100% (2) | 100% (2) |
| CREDIT_CARD | 100% (2) | 100% (2) | 100% (2) |
| DOB | 67% (3) | 100% (3) | 100% (3) |
| SALARY_INR | 75% (4) | 0% (4) | 0% (4) |
| PERSON | 88% (8) | 88% (8) | 88% (8) |
| **precision** | 93.8% | 54.3% | 58.1% |
| **recall** | 85.7% | 54.3% | 71.4% |
| **f1** | 89.5% | 54.3% | 64.1% |
