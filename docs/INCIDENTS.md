# Incident analyses (Objective O4)

Three documented incidents involving LLM assistants in organisational settings, each traced through the data path
of Review 2 Fig. 1 and replayed against Copilot Privacy Guard's controls. Facts are limited to what the cited sources
report. Measured numbers refer to `results/` (seed 42).

---

## 1. EchoLeak — Microsoft 365 Copilot (CVE-2025-32711, June 2025)

**What happened.** Researchers at Aim Security reported a zero-click flaw in Microsoft 365 Copilot, tracked as
CVE-2025-32711 and rated critical (CVSS 9.3). Microsoft described it as "AI command injection in M365 Copilot allows an
unauthorized attacker to disclose information over a network". An attacker could send a crafted email; when Copilot
later retrieved that email as context for an ordinary user question, the injected instructions could make Copilot leak
sensitive data from its context to the attacker without any user interaction. Aim Security characterised it as an
"LLM scope violation" enabling indirect prompt injection. Microsoft fixed it server-side in the June 2025 Patch Tuesday;
no in-the-wild exploitation was reported.

**Root-cause chain.**
1. Attacker sends an email whose body contains instructions aimed at the assistant (untrusted, external content).
2. The email is ingested into the tenant's searchable corpus (Fig. 1: *cross-silo ingestion*).
3. A user asks Copilot a routine question; retrieval selects the attacker's email alongside internal documents (*retrieval*).
4. The model treats the email's text as instructions rather than data (*inference*) — the scope violation.
5. The model emits sensitive context in a form that leaves the tenant (an outbound link/image request).

**Exposure point.** Retrieval → inference: untrusted retrieved content steering the model.

| Control | Would it have stopped it? | Why |
|---|---|---|
| Injection screening (heuristic + LLM judge) | **Partially** | Our judge flags paraphrased and non-English instructions the heuristic misses (see `results/injection_bench.md`); a novel phrasing can still pass. Surviving chunks are wrapped as `<untrusted_document>`. |
| PII redaction (prompt, index, chunks) | **Partially** | Personal identifiers in the context would be placeholders, so an exfiltrated answer carries `<AADHAAR_1>` rather than the value. Non-personal confidential text is not redacted. |
| Purpose-based access filter | **No** | The email sits in the user's own mailbox; the user is entitled to it. |
| Privacy-preserving index | **No** | Not an index-inversion attack. |
| Hash-chained audit | **Yes, for forensics** | Every retrieval, drop and judge score is recorded tamper-evidently, supporting DPDP §8(6) breach intimation. |

**Residual gap.** An egress control on model output (strip or allow-list outbound links/images) — not in this prototype.

Sources: The Hacker News, "Zero-Click AI Vulnerability Exposes Microsoft 365 Copilot Data Without User Interaction"
(June 2025) — https://thehackernews.com/2025/06/zero-click-ai-vulnerability-exposes.html ; SOC Prime, "CVE-2025-32711
Vulnerability: EchoLeak" — https://socprime.com/blog/cve-2025-32711-zero-click-ai-vulnerability/

---

## 2. Slack AI — indirect prompt injection across channels (August 2024)

**What happened.** PromptArmor showed that Slack AI, which searches public channels (including ones the user has not
joined) as well as the user's private channels, could be steered by an instruction planted in a public channel. In
their demonstration, a developer's API key posted in a private channel was exfiltrated: the planted instruction made
Slack AI render a Markdown link that carried the private data to an attacker-controlled server in the query string.
PromptArmor disclosed on 14 August 2024; Slack's security team initially concluded the behaviour was "intended"
because public-channel messages are searchable by design, and PromptArmor published on 20 August 2024.

**Root-cause chain.**
1. Attacker posts an instruction in a public channel (no access to the victim's private channel needed).
2. Slack AI's retrieval spans public and private channels in one context (*cross-silo retrieval*).
3. The victim asks about their own private data; both the secret and the attacker's instruction are retrieved.
4. The model follows the planted instruction and builds an exfiltration link containing the secret (*inference*).
5. The victim clicks the rendered link, sending the secret to the attacker.

**Exposure point.** Retrieval mixing trust zones (public, attacker-writable content + private secrets) → inference.

| Control | Would it have stopped it? | Why |
|---|---|---|
| Injection screening | **Yes, likely** | The planted text is an explicit instruction to the assistant — the class our heuristic and judge both target; it is dropped before the model sees it. |
| PII redaction | **Partially** | An API key is a secret, not personal data; our recognizers do not cover credential formats. |
| Purpose-based access filter | **Partially** | A purpose tag on the private channel would not block the victim's own access; it would stop cross-purpose mixing only if channels were tagged. |
| Hash-chained audit | **Yes, for forensics** | Records which public-channel message was retrieved and whether it was flagged. |

**Residual gap.** Credential/secret recognizers and an output link allow-list.

Sources: PromptArmor, "Data Exfiltration from Slack AI via indirect prompt injection" —
https://promptarmor.com/resources/data-exfiltration-from-slack-ai-via-indirect-prompt-injection ; The Register
(21 Aug 2024) — https://www.theregister.com/2024/08/21/slack_ai_prompt_injection/ ; Dark Reading —
https://darkreading.com/cyberattacks-data-breaches/slack-ai-patches-bug-that-let-attackers-steal-data-from-private-channels

---

## 3. Samsung — confidential source code sent to ChatGPT (2023)

**What happened.** Samsung engineers uploaded internal source code to ChatGPT (reportedly to check it for errors).
An internal memo (dated 28 April 2023 per KED Global) restricted generative-AI tools — initially for the Device
eXperience division — on company devices and networks while security measures were reviewed, citing the risk that
data sent to external AI services is stored outside the company's control. An internal survey cited in coverage found
65% of respondents saw such services as a security risk.

**Root-cause chain.**
1. An employee pastes confidential material into a prompt (*prompt capture*).
2. The prompt leaves the organisation for a third-party LLM provider (*model inference at an external processor*).
3. The provider's logging/telemetry retains it outside the organisation's control (*telemetry & logging*).

**Exposure point.** Prompt capture → external processor — exactly the "context PII exposure" this study measures.

| Control | Would it have stopped it? | Why |
|---|---|---|
| PII redaction of the prompt | **Partially** | Personal data in the prompt (names, emails, phone numbers in code comments or configs) is replaced before it leaves; the source code itself is not personal data and passes. Our measured context PII exposure under `full` is 0% on the evaluation set. |
| Access / injection / index | **No** | Not an access, injection or index problem. |
| Hash-chained audit | **Yes, for accountability** | Records that a prompt was sent and which entity types were redacted, without storing the values (DPDP Rules r.6 logging). |

**Residual gap.** Trade-secret / source-code classification before egress (DLP for non-personal confidential data).

Sources: CSO Online, "Samsung bans staff AI use over data leak concerns" —
https://www.csoonline.com/article/3695170/samsung-bans-staff-ai-use-over-data-leak-concerns.html ; Business Today
(2 May 2023) — https://www.businesstoday.in/technology/news/story/samsung-bans-the-use-of-chatgpt-google-bard-bing-chatbot-and-other-ai-tools-379661-2023-05-02 ;
KED Global — https://www.kedglobal.com/newsView/ked202305020022

---

## Cross-incident summary

| Incident | Exposure point (Fig. 1) | Injection screening | PII redaction | Access filter | Audit | Main residual gap |
|---|---|---|---|---|---|---|
| EchoLeak (2025) | retrieval → inference | Partially | Partially | No | Yes | output egress / link allow-list |
| Slack AI (2024) | cross-zone retrieval → inference | Yes (likely) | Partially | Partially | Yes | secret recognizers, link allow-list |
| Samsung (2023) | prompt capture → external processor | No | Partially | No | Yes | non-personal confidential-data classification |

**Lesson for the design.** Two of three incidents are injection-through-retrieval, which is why Control 3 combines a
heuristic with an LLM judge; all three end with data leaving through the model's input or output channel, which is why
the study measures *context* exposure to the provider, not only answer leakage. None is fully stopped by a single
control — the argument for layered, audited middleware rather than any one defence.
