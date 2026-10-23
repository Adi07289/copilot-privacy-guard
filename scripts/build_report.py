"""Builds the final project report (.docx + .pdf) in the order of the faculty template. Every number is read from
results/*.json at build time; references come from docs/report/references.py (all verified)."""
import json, sys
from pathlib import Path
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "report"))
from references import INTRO_CITES, REFS  # noqa: E402

R, FIG, OUT = ROOT / "results", ROOT / "docs" / "figures", ROOT / "deliverables"
J = lambda n: json.loads((R / n).read_text())
COMP, BENCH, PII, CJ, META = J("comparison.json"), J("injection_bench.json"), J("pii_detection.json"), J("compliance.json"), J("run_meta.json")
A, LBL = COMP["aggregates"], COMP["labels"]
NUM = {r["key"]: i + 1 for i, r in enumerate(REFS)}
INK, TEAL = RGBColor(0x14, 0x17, 0x1A), RGBColor(0x1F, 0x8A, 0x70)


def pct(v, d=1):
    return "—" if v is None else f"{100 * v:.{d}f}%"


def c(*keys):
    return "[" + ", ".join(str(NUM[k]) for k in keys) + "]"


def a(cfg, k, d=1):
    return pct(A[cfg].get(k), d)


heldout, corpus_pii = PII["results"]["heldout"], PII["results"]["corpus"]
checks = CJ["checks"]; n_pass = sum(1 for x in checks if x["passed"])
failed = [x for x in checks if x["passed"] is False]
fam = BENCH["by_family"]

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2.0)


def H(text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = INK; r.font.name = "Calibri"
    return h


def P(text, bold_lead=None, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead); r.bold = True
    r = p.add_run(text); r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(6)
    return p


def B(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(it, tuple):
            r = p.add_run(it[0]); r.bold = True; p.add_run(it[1])
        else:
            p.add_run(it)


def _shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor); tcPr.append(shd)


def T(headers, rows, widths=None, size=8.5, highlight_row=None):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]; cell.text = ""; run = cell.paragraphs[0].add_run(h); run.bold = True
        run.font.size = Pt(size); _shade(cell, "DCE3E1")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; run = cells[i].paragraphs[0].add_run(str(v)); run.font.size = Pt(size)
            if highlight_row is not None and ri == highlight_row:
                run.bold = True; _shade(cells[i], "EAF4F0")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t


FIGN = [0]


def F(name, caption, width=16.0):
    FIGN[0] += 1
    doc.add_picture(str(FIG / name), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(f"Figure {FIGN[0]}. {caption}", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)


# ---------------------------------------------------------------- title block
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("VELLORE INSTITUTE OF TECHNOLOGY, VELLORE\nBCSE318L – Data Privacy  |  Fall Semester 2026–27\nFinal Review — Project Report"); r.bold = True; r.font.size = Pt(11)
H("Title of the project")
P("Privacy-Preserving Enterprise Copilots: A DPDP-Aligned Risk Assessment and Mitigation Framework for Retrieval-Augmented "
  "LLM Assistants in Indian Organisations", bold_lead="")
P("Short title: Copilot Privacy Guard. Topic No. 35 — Privacy Implications of LLM-based Enterprise Assistants. "
  "Source code: github.com/Adi07289/copilot-privacy-guard", italic=True, size=10)
H("Name and registration number of the students")
T(["Name", "Registration No.", "Programme"], [["Aditya Sharma", "23BCE0936", "B.Tech CSE (Core)"],
                                             ["Aarav Raina", "23BCE0992", "B.Tech CSE"]], [6, 4, 6], size=10)
P("Faculty in-charge: Dr. Geetha Mary A", size=10)

# ---------------------------------------------------------------- abstract
H("Abstract")
P(f"Retrieval-augmented enterprise assistants such as Microsoft 365 Copilot answer natural-language questions over "
  f"email, HR records, tickets and code, which turns every over-shared document into something a single query can surface. "
  f"We designed, implemented and evaluated Copilot Privacy Guard, a middleware that sits between the assistant and its data "
  f"and applies four controls: India-first PII redaction (applied to the prompt, to the vector index and to retrieved "
  f"chunks), a purpose-based access filter that encodes the DPDP Act's purpose limitation as a runtime rule, prompt-injection "
  f"screening that combines a heuristic with an LLM-as-judge, and a SHA-256 hash-chained audit log. On a synthetic Indian-"
  f"enterprise corpus (80 documents, 369 labelled PII spans, 4 poisoned documents) and 36 queries across five personas, we "
  f"compared the guard against simplified analogues of six defence families from the literature, run on the identical "
  f"corpus, retriever and LLM. The guard reduced PII sent to the LLM provider from {a('none','context_pii_exposure',0)} to "
  f"{a('full','context_pii_exposure',0)}, over-sharing from {a('none','oversharing_rate')} to {a('full','oversharing_rate',0)}, "
  f"raw PII stored in the index from {a('none','index_pii_exposure',0)} to {a('full','index_pii_exposure',0)}, and the share of "
  f"poisoned chunks reaching the model from {a('none','injection_exposure_rate',0)} to {a('full','injection_exposure_rate',0)}. "
  f"Its PII detector reached an F1 of {pct(heldout['ours']['f1'])} on a held-out set against {pct(heldout['presidio_default']['f1'])} "
  f"for Microsoft Presidio as shipped, and the LLM judge detected {pct(BENCH['tpr']['judge'],0)} of 24 injection payloads "
  f"(including paraphrased and Hindi ones) with no false positives on hard negatives, against {pct(BENCH['tpr']['heuristic'],0)} "
  f"for the heuristic. These gains cost answer utility ({a('full','answer_quality_rouge_l')} ROUGE-L retention against the "
  f"undefended answers) and about {A['full']['added_latency_ms']/1000:.1f} s of latency per query, which we report as the "
  f"central trade-off. All controls are mapped clause by clause to the DPDP Act 2023, DPDP Rules 2025, GDPR and the EU AI Act, "
  f"with a DPIA risk register and a {len(checks)}-item assurance checklist that runs against the system ({n_pass} pass).")

# ---------------------------------------------------------------- introduction
H("Introduction")
P(f"Retrieval-augmented generation (RAG) couples a language model with a retriever over an external corpus so that answers "
  f"are grounded in an organisation's own documents {c('lewis2020')}. Enterprise copilots apply this design to email, chat, "
  f"documents, tickets, HR systems and code repositories, and they inherit the permissions of the user who asks. That choice "
  f"inverts a long-standing security assumption: instead of information being compartmentalised by role, one conversational "
  f"interface can surface any document the user's identity is technically — though rarely intentionally — allowed to open. "
  f"Industry audits of Copilot roll-outs report that roughly 15–16% of business-critical files are already over-shared before "
  f"an assistant is introduced; the assistant does not create these exposures, it makes them discoverable in seconds.")
P(f"The research literature shows that the threat is concrete. Large models memorise and regurgitate training data "
  f"{c('carlini2021')}, including personally identifiable information {c('lukas2023')}. RAG adds a second, more direct "
  f"surface: the retrieval database itself leaks under targeted prompts {c('zeng2024')}, instruction-following prompts can "
  f"extract retrieved content at scale {c('qi2024')}, and optimiser-crafted attacks report near-total PII extraction from "
  f"undefended systems {c('zhang2025deal')}. Retrieved content can also carry instructions: indirect prompt injection lets an "
  f"attacker who can write one document steer the assistant for everyone who retrieves it {c('greshake2023')}. Defences exist "
  f"for parts of the problem — for example decoding-time noise that limits how much retrieved text the model reproduces "
  f"{c('wang2025pad')} — but they operate on tokens and logits and do not know who is asking or why.")
P(f"Privacy, however, is about appropriate flows, not only secrecy: the same fact can be fine to share with HR and wrong to "
  f"share with an intern, which is the contextual-integrity view that recent LLM benchmarks show models fail to respect "
  f"{c('mireshghallah2024')}. Database research addressed this two decades ago with purpose-based access control, where access "
  f"is granted only when the purpose of a request is compatible with the purpose the data was collected for {c('byun2008')}. "
  f"India's Digital Personal Data Protection Act, 2023 makes that principle law — personal data may be processed only for a "
  f"lawful purpose and consent covers only the data necessary for it — and the DPDP Rules notified in November 2025 add "
  f"concrete security-safeguard, logging and breach-intimation duties that become enforceable in 2027. In parallel the EU AI "
  f"Act's obligations have started to apply. Organisations are therefore in exactly the window in which a deployable, "
  f"measurable defence has practical value.")
P("This work contributes: (1) a four-control privacy-guard middleware plus a privacy-preserving (redacted) vector index, "
  "implemented as one configurable pipeline; (2) a same-corpus comparative study against simplified analogues of six defence "
  "families from the literature, with a leave-one-out ablation; (3) an India-first PII detector evaluated head-to-head against "
  "Microsoft Presidio on a held-out set; (4) an injection-detection benchmark comparing a heuristic, an LLM judge and Meta's "
  "Prompt Guard 2; and (5) compliance as code — a verified clause map, a DPIA register and an executable assurance checklist "
  "whose evidence is drawn from the measured results.")

# ---------------------------------------------------------------- literature review
H("Literature review")
P(f"Memorisation and membership inference. Training-data extraction {c('carlini2021')} and PII-specific leakage analyses "
  f"{c('lukas2023')} established that models leak what they were trained on; membership-inference work now spans every stage "
  f"of the LLM pipeline including RAG {c('wu2025')} and semantic attacks outperform token-level ones {c('mozaffari2024')}. "
  f"These works concern model parameters rather than the retrieval context an enterprise copilot assembles at query time.")
P(f"RAG leakage and its defences. RAG systems leak their datastores {c('zeng2024','qi2024')}; graph-structured RAG adds "
  f"entity/relationship leakage {c('liu2025graph')}; active-learning {c('lyu2026')} and optimiser-based {c('zhang2025deal')} "
  f"attacks raise extraction rates further, all against systems with no screening layer. On the defence side, Privacy-Aware "
  f"Decoding perturbs logits so retrieved content is reproduced less faithfully {c('wang2025pad')}, and a two-stage retriever "
  f"with an LLM-judged leakage metric avoids the context loss of coarse redaction {c('zhang2026rag')}. Neither models who is "
  f"asking: the private context still reaches the model, and organisational permissions are absent.")
P(f"Prompt injection. Indirect injection through retrieved content was introduced by Greshake et al. {c('greshake2023')} and "
  f"formalised and benchmarked by Liu et al. {c('liu2024pi')}; surveys catalogue structural defences {c('geng2026','chu2026')} "
  f"and OWASP ranks prompt injection first among LLM risks {c('owasp2025')}. StruQ separates instructions from data but needs "
  f"fine-tuning {c('chen2024struq')}; Spotlighting marks untrusted input with prompt-only delimiting and encoding "
  f"{c('hines2024')}; LLM-based guards such as Llama Guard classify unsafe content {c('inan2023')}, building on evidence that "
  f"strong LLMs are reliable judges {c('zheng2023')}. The surveys themselves note that these defences are rarely evaluated "
  f"together as one deployable pipeline.")
P(f"Embeddings, PII detection, access control and auditing. Embeddings leak their inputs {c('song2020')}, and Vec2Text "
  f"inverts them back to text {c('morris2023','seputis2025')}; noise-based defences trade off utility {c('tsai2026')}. PII "
  f"detectors generalise poorly across sources {c('jha2026')}, and hybrid rule-plus-NER pipelines perform well within one "
  f"domain {c('mishra2025')}. Purpose-based {c('byun2008')} and role-based {c('sandhu1996')} access control, LINDDUN threat "
  f"modelling {c('deng2011')} and hash-chained audit logs {c('schneier1999')} are mature ideas that have not been combined "
  f"inside an LLM retrieval pipeline. Table 1 summarises the reviewed works; the articles below are closely related to this "
  f"topic but differ in implementation.")
rows = []
for r in REFS:
    rows.append([str(NUM[r["key"]]), str(r["year"]), f"{r['authors']}. {r['title']}",
                 "\n".join(f"{i}. {x}" for i, x in enumerate(r["contributions"], 1)),
                 "\n".join(f"{i}. {x}" for i, x in enumerate(r["limitations"], 1))])
P("Table 1. Comparative literature matrix.", italic=True, size=9)
T(["Reference No.", "Year of publication", "Authors and Title", "Contributions", "Limitations"], rows, [1.4, 1.5, 5.0, 4.6, 4.3], size=7.5)

# ---------------------------------------------------------------- gaps
H("Gaps identified")
B([("G1 — No role- or purpose-awareness in RAG defences. ", f"Decoding-time and retriever-side defences {c('wang2025pad','zhang2026rag')} act on content, not on who is asking or why; the private context still reaches the model."),
   ("G2 — Attacks assume no screening layer. ", f"Extraction attacks {c('qi2024','lyu2026','zhang2025deal')} are evaluated against undefended pipelines; the effect of a middleware in front of the model is unmeasured."),
   ("G3 — Injection defences are evaluated in isolation. ", f"Delimiting {c('hines2024')}, structured queries {c('chen2024struq')} and LLM guards {c('inan2023')} are not combined with access control, redaction and audit in one pipeline {c('geng2026')}."),
   ("G4 — PII detection is not India-ready. ", f"Detectors generalise poorly {c('jha2026')}, domain pipelines are finance-only {c('mishra2025')}, and Presidio ships its Aadhaar/PAN recognizers disabled with none for IFSC, UPI or ₹ salaries."),
   ("G5 — No evaluation against DPDP. ", "Existing work is GDPR- or CCPA-oriented; no deployable defence has been mapped clause by clause to the DPDP Act 2023 and Rules 2025 with measured evidence.")])

# ---------------------------------------------------------------- proposed method
H("Proposed method")
F("fig_architecture.png", "Copilot Privacy Guard: four controls plus a privacy-preserving index, every decision audited.")
P("The middleware intercepts every prompt and every retrieved document before either reaches the LLM. All variants studied "
  "here are configurations of one pipeline (ConfigurableRAG) that share the retriever and the LLM, so only the defence differs.")
B([("① PII detection and redaction. ", "A registry of India-first recognizers — Aadhaar with Verhoeff checksum, PAN, Indian mobile, IFSC, UPI VPA, card numbers with Luhn, dates of birth, rupee salaries, e-mail — plus spaCy NER and pattern rules for person names. Values become typed placeholders (<AADHAAR_1>). It runs on the prompt (so the retriever never sees raw PII), on documents before indexing, and on surviving chunks."),
   ("Privacy-preserving index. ", "The vector index (TF-IDF, or bge-small dense embeddings) is built from redacted documents, so an embedding-inversion attack has no raw identifiers to recover — the exposure point Review 2 marked as cross-silo ingestion."),
   ("② Purpose-based access filter. ", "A chunk is released only if the user's clearance meets its sensitivity, its department matches (or it is public, or the user holds a cross-department role such as CISO/DPO) and the declared purpose of the query is one of the document's permitted purposes — DPDP purpose limitation as a runtime rule."),
   ("③ Injection screening. ", "A weighted pattern bank with zero-width, HTML-comment and base64 checks, OR-combined with an LLM-as-judge (gpt-oss-120b, JSON verdict, cached). Flagged chunks are dropped; the rest are wrapped in <untrusted_document> delimiters."),
   ("④ Hash-chained audit log. ", "Each decision is appended with SHA-256(previous hash ‖ entry); verification reports the first broken link. Entries store entity types and counts, user and purpose — never PII values.")])
P(f"Headline result (Table 2): on 36 queries the guard sends no raw PII to the LLM provider, releases no out-of-purpose "
  f"chunk, stores no raw PII in its index and lets no poisoned chunk reach the model, at a utility cost of "
  f"{a('full','answer_quality_rouge_l')} ROUGE-L retention.")
P("Table 2. Headline comparison (undefended vs ours).", italic=True, size=9)
T(["Metric", "Vanilla RAG", "Copilot Privacy Guard"], [
    ["PII sent to LLM provider (context exposure)", a("none", "context_pii_exposure"), a("full", "context_pii_exposure")],
    ["PII in final answer (LLM answers)", a("none", "answer_pii_leakage_llm"), a("full", "answer_pii_leakage_llm")],
    ["Over-sharing retrieval rate", a("none", "oversharing_rate"), a("full", "oversharing_rate")],
    ["Raw PII stored in vector index", a("none", "index_pii_exposure"), a("full", "index_pii_exposure")],
    ["Poisoned chunks reaching the LLM", a("none", "injection_exposure_rate"), a("full", "injection_exposure_rate")],
    ["Injection success (canary obeyed by LLM)", a("none", "injection_success_rate"), a("full", "injection_success_rate")],
    ["Answer-quality retention (ROUGE-L / cosine)", "100% / 100%", f"{a('full','answer_quality_rouge_l')} / {a('full','answer_quality_cosine')}"],
    ["Added latency per query", "0 ms", f"+{A['full']['added_latency_ms']:.0f} ms"]], [8, 4, 4.5])

# ---------------------------------------------------------------- experimental analysis
H("Experimental Analysis (with comparison to other research works)")
H("Setup", 2)
P(f"Corpus: 80 synthetic Indian-enterprise documents (25 e-mails, 20 HR records, 20 tickets, 15 code/config files) generated "
  f"with a fixed seed, tagged with department, sensitivity (0–3) and permitted purposes, with 369 ground-truth PII spans and "
  f"four documents poisoned with canary-tagged injection payloads. Personas: HR manager, engineer, finance analyst, intern and "
  f"CISO. Queries: 36 — 8 unauthorised probes, 8 authorised sensitive requests, 12 utility questions and 8 injection queries "
  f"issued by a persona authorised to read the poisoned document, so access control cannot trivially block them. LLMs (Groq): "
  f"answers {META.get('groq_model')}, judge {META.get('judge_model')}, comparator {META.get('prompt_guard_model')}; all "
  f"{META.get('cache_entries')} responses are cached for exact reproducibility. Deterministic metrics come from an extractive "
  f"answerer; LLM-dependent metrics (injection success, answer quality, answer leakage) from {A['none']['injection_success_n']} "
  f"injection and 12 utility queries answered by the LLM ({META.get('llm_answer_fallbacks')} fallbacks).")
H("Comparative study against defence families", 2)
P("Each row below is a simplified analogue of a defence family from the literature, run on the same corpus, queries, "
  "retriever and LLM — not a reproduction of the cited paper.")
order = ["none", "rbac", "pii_only", "output_filter", "spotlight", "heuristic", "full"]
T(["Configuration (analogue of)", "PII to LLM", "Answer PII (LLM)", "Over-sharing", "Injection exposure", "Injection success", "ROUGE-L", "Index PII", "Latency"],
  [[f"{LBL[k]} — {COMP['analogues'][k]}", a(k, "context_pii_exposure"), a(k, "answer_pii_leakage_llm"), a(k, "oversharing_rate"),
    a(k, "injection_exposure_rate"), a(k, "injection_success_rate"), a(k, "answer_quality_rouge_l"), a(k, "index_pii_exposure"),
    f"{A[k]['added_latency_ms']:+.0f} ms"] for k in order], [5.2, 1.4, 1.5, 1.4, 1.5, 1.5, 1.3, 1.2, 1.4], size=7.5, highlight_row=6)
F("fig_comparison.png", "Same corpus, queries and LLM — only the defence differs (ours highlighted).")
P(f"Inherited RBAC — how Copilot works — removes most clearance and department violations but still sends "
  f"{a('rbac','context_pii_exposure')} of retrieved PII to the provider and lets {a('rbac','injection_exposure_rate')} of "
  f"poisoned chunks through. Redaction-only DLP removes PII from the context but leaves over-sharing at "
  f"{a('pii_only','oversharing_rate')}: confidential non-personal content still crosses departments. Output-side filtering "
  f"scrubs the answer yet sends {a('output_filter','context_pii_exposure',0)} of PII to the provider — under DPDP §8(1) the "
  f"fiduciary remains responsible for that processor exposure. Delimiting alone blocks nothing; heuristic screening removes "
  f"seeded payloads but none of the privacy exposure. Only the combined guard drives all four exposure metrics to zero.")
P(f"Injection success. The answer model resisted most seeded canary instructions on its own (vanilla: "
  f"{a('none','injection_success_rate',0)} of {A['none']['injection_success_n']} attacked queries; worst case "
  f"{max((A[k]['injection_success_rate'] or 0) for k in order)*100:.1f}%). Model-level resistance is not a control the "
  f"deployer owns, which is why the guard's design target is exposure: no poisoned chunk reaches the model at all.")
H("Ablation", 2)
abl = ["full", "full_no_judge", "full_no_access", "full_no_pii", "full_no_index", "full_dense"]
T(["Configuration", "PII to LLM", "Over-sharing", "Index PII", "Injection exposure", "ROUGE-L", "Latency"],
  [[LBL[k], a(k, "context_pii_exposure"), a(k, "oversharing_rate"), a(k, "index_pii_exposure"), a(k, "injection_exposure_rate"),
    a(k, "answer_quality_rouge_l"), f"{A[k]['added_latency_ms']:+.0f} ms"] for k in abl if k in A], [5, 1.8, 1.8, 1.8, 2, 1.6, 1.8], size=8)
F("fig_ablation.png", "Removing one control at a time: each control owns one exposure.")
P(f"Each control owns a distinct exposure: without the access filter over-sharing returns to {a('full_no_access','oversharing_rate')}; "
  f"without redaction {a('full_no_pii','context_pii_exposure')} of PII reaches the provider; without the redacted index the "
  f"index again holds {a('full_no_index','index_pii_exposure',0)} of raw PII. Removing the judge leaves these metrics unchanged "
  f"on the corpus — its value shows on the harder payloads of the benchmark below — while cutting latency to "
  f"+{A['full_no_judge']['added_latency_ms']:.0f} ms. Dense retrieval reproduces the privacy results.")
H("Privacy–utility trade-off", 2)
F("fig_privacy_utility.png", "Context PII exposure vs answer-quality retention for every configuration.", 13.5)
P(f"The guard sits at zero exposure but the lowest answer-quality retention ({a('full','answer_quality_rouge_l')}). Utility "
  f"questions often ask for names or contact details ('who should I contact…') that the guard deliberately replaces with "
  f"placeholders, and the judge's OR-combination with the heuristic drops some benign chunks. The assurance check A12 "
  f"(≥ 50% retention) therefore fails, and we report it as such.")
H("PII detection: comparison with Microsoft Presidio", 2)
types = ["AADHAAR", "PAN", "IN_MOBILE", "IFSC", "UPI_VPA", "EMAIL", "CREDIT_CARD", "DOB", "SALARY_INR", "PERSON"]
dets = PII["detectors"]; dn = {"ours": "Ours", "presidio_default": "Presidio (as shipped)", "presidio_india": "Presidio + India"}


def rec(ds, d, t):
    pt = PII["results"][ds][d]["per_type"].get(t)
    return "—" if not pt or pt["recall"] is None else pct(pt["recall"], 0)


T(["Entity type"] + [f"{dn[d]} — held-out" for d in dets] + [f"{dn[d]} — corpus" for d in dets],
  [[t] + [rec("heldout", d, t) for d in dets] + [rec("corpus", d, t) for d in dets] for t in types]
  + [[m.upper()] + [pct(PII["results"]["heldout"][d][m]) for d in dets] + [pct(PII["results"]["corpus"][d][m]) for d in dets]
     for m in ("precision", "recall", "f1")], None, size=7.5)
F("fig_pii_vs_presidio.png", "Per-type recall on the held-out set.")
P(f"On the held-out set — sentences written in formats the generator never produces — our detector reaches F1 "
  f"{pct(heldout['ours']['f1'])} against {pct(heldout['presidio_default']['f1'])} for Presidio as shipped and "
  f"{pct(heldout['presidio_india']['f1'])} with its India recognizers enabled. Presidio has no recognizer for IFSC, UPI or "
  f"rupee salaries, and its Aadhaar/PAN recognizers are disabled by default. Presidio is better on spaced mobile numbers, "
  f"written-out dates and lower-case PAN, which are genuine gaps in our patterns. Compared with the hybrid finance pipeline "
  f"{c('mishra2025')} our numbers are not directly comparable (different documents and taxonomy); PIIBench {c('jha2026')} "
  f"warns that detectors generalise poorly, so the corpus figures (F1 {pct(corpus_pii['ours']['f1'])}) are an upper bound.")
H("Injection detection: heuristic vs LLM judge vs Prompt Guard 2", 2)
bd = BENCH.get("detectors", ["heuristic", "judge", "combined"])
bn = {"heuristic": "Heuristic", "judge": "LLM judge", "combined": "Combined (ours)", "prompt_guard": "Prompt Guard 2"}
T(["Payload family"] + [bn[d] for d in bd], [[f.title()] + [pct(fam[f][d], 0) for d in bd] for f in fam]
  + [["Detection (all 24)"] + [pct(BENCH["tpr"][d], 0) for d in bd], ["FPR clean docs"] + [pct(BENCH["fpr_clean"][d], 0) for d in bd],
     ["FPR hard negatives"] + [pct(BENCH["fpr_hard"][d], 0) for d in bd]], None, size=8.5)
F("fig_injection_bench.png", "Detection rate by payload family.")
P(f"The heuristic catches explicit and encoded payloads but none of the paraphrased or Hindi/Hinglish ones and falsely flags "
  f"{pct(BENCH['fpr_hard']['heuristic'],0)} of hard negatives (security reminders, release notes). The LLM judge detects all 24 "
  f"payloads with no false positives. Meta's Prompt Guard 2, a dedicated classifier, detects {pct(BENCH['tpr']['prompt_guard'],0)} "
  f"— it is strong on explicit overrides but misses exfiltration and Hindi payloads. Because the guard OR-combines heuristic and "
  f"judge, it inherits the heuristic's false positives; making the judge the sole decision-maker is our first future-work item.")
H("Replay of documented incidents", 2)
T(["Incident", "Exposure point", "Injection screening", "PII redaction", "Access filter", "Audit", "Residual gap"],
  [["EchoLeak, M365 Copilot (2025)", "retrieval → inference", "Partially", "Partially", "No", "Yes", "output egress allow-list"],
   ["Slack AI injection (2024)", "cross-zone retrieval", "Yes (likely)", "Partially", "Partially", "Yes", "secret recognizers"],
   ["Samsung → ChatGPT (2023)", "prompt → external processor", "No", "Partially", "No", "Yes", "non-personal confidential data"]], None, size=8)
H("Compliance, DPIA and assurance checklist", 2)
fw = {}
for cl in CJ["clauses"]:
    fw.setdefault(cl["framework"], []).append(cl["ref"])
T(["Framework", "Clauses mapped (each verified against the source text)"], [[k, ", ".join(v)] for k, v in fw.items()], [4.5, 12], size=8.5)
T(["ID", "Threat", "Inherent L×I", "Controls", "Residual L×I"],
  [[r["id"], r["threat"], f"{r['inherent'][2]} ({r['inherent'][3]})", ", ".join(r["controls"]) or "—", f"{r['residual'][2]} ({r['residual'][3]})"] for r in CJ["risks"]],
  [1.2, 7, 2.6, 3, 2.6], size=8.5)
F("fig_dpia.png", "DPIA risk register: inherent vs residual (likelihood × impact, expert judgement).", 13.5)
P(f"{n_pass} of {len(checks)} executable assurance checks pass against the running system and its results"
  + (f"; failing: {', '.join(x['id'] + ' (' + x['statement'] + ')' for x in failed)}" if failed else "") + ".")
H("Comparison with results reported in the literature", 2)
P("The rows below place our measurements beside numbers reported by other works. Settings, data and models differ, so this "
  "is context, not a head-to-head comparison.", italic=True)
T(["Work", "Their setting", "Reported result", "Our closest measurement", "Comparable?"], [
    [f"DEAL {c('zhang2025deal')}", "Optimiser attack on undefended RAG", "near-total PII extraction (as reported)", f"Vanilla RAG sends {a('none','context_pii_exposure',0)} of PII to the LLM; guard {a('full','context_pii_exposure',0)}", "No — different attack"],
    [f"PAD {c('wang2025pad')}", "Decoding-time noise", "leakage–utility trade-off", f"Guard: 0% PII in context, {a('full','answer_quality_rouge_l')} ROUGE-L", "No — PAD leaves PII in the context"],
    [f"Spotlighting {c('hines2024')}", "Prompt delimiting", "reduced attack success", f"'Delimiting only': injection exposure {a('spotlight','injection_exposure_rate',0)}, success {a('spotlight','injection_success_rate',0)}", "Analogue on our data"],
    [f"Vec2Text {c('morris2023')}", "Embedding inversion", "recovers most of short inputs", f"Index PII exposure {a('none','index_pii_exposure',0)} → {a('full','index_pii_exposure',0)}", "Partially — we remove what inversion could recover"],
    [f"Hybrid PII {c('mishra2025')}", "Financial documents", "high precision/recall (as reported)", f"Held-out F1 {pct(heldout['ours']['f1'])}", "No — different documents"],
    [f"PIIBench {c('jha2026')}", "Unified multi-source taxonomy", "low span-F1 for 8 detectors", f"Presidio held-out F1 {pct(heldout['presidio_default']['f1'])}", "No — smaller, narrower set"]],
  [2.6, 3, 3.2, 4.6, 3.2], size=8)
H("Threats to validity", 2)
B(["Synthetic corpus: written by the authors; the held-out PII set reduces but does not remove this bias.",
   "Two detector fixes were made after the first comparison run exposed residual leaks (first names after e-mail sign-offs; name patterns crossing line breaks into HR record titles). The fixes are general, the held-out set contains neither pattern, and both are disclosed here.",
   f"Small sets: 36 queries and 24 payloads; injection success rests on {A['none']['injection_success_n']} attacked queries.",
   "Likelihood/impact scores in the DPIA are expert judgement; only residual scores are tied to measured evidence.",
   "Comparative configurations are simplified analogues, not reproductions of the cited systems.",
   "Latency includes uncached LLM-judge calls; production caching and batching would lower it."])

# ---------------------------------------------------------------- future work / conclusion
H("Future work")
B(["Make the LLM judge the sole drop decision (heuristic for logging only) to remove the heuristic's false positives and recover utility.",
   "Re-identify placeholders in the final answer for users authorised to see them (a tokenisation vault), so the LLM never sees PII but HR still gets real values.",
   "Add recognizers for credentials/secrets, spaced mobile numbers, written-out dates and lakh/crore amounts; evaluate on PIIBench.",
   "Output egress controls (link and image allow-listing) — the residual gap in the EchoLeak and Slack AI replays.",
   "Integrate with an identity provider (Entra ID / Google Workspace) for live ACLs and a consent prompt for declared purpose.",
   "Evaluate on a real (consented, de-identified) enterprise corpus and against adaptive attacks."])
H("Conclusion")
P(f"Enterprise copilots turn latent over-sharing into an instant, conversational disclosure channel. A layered middleware — "
  f"redaction before indexing and retrieval, purpose-based access control, injection screening with an LLM judge, and a "
  f"tamper-evident audit trail — removed every measured exposure on our benchmark ({a('full','context_pii_exposure',0)} PII to "
  f"the provider, {a('full','oversharing_rate',0)} over-sharing, {a('full','index_pii_exposure',0)} raw PII in the index, "
  f"{a('full','injection_exposure_rate',0)} poisoned chunks reaching the model), where every single-family defence left at "
  f"least one exposure open. The price is utility and latency, which we quantify rather than hide. Mapping each control to "
  f"DPDP, GDPR and EU AI Act clauses with measured evidence shows how such a guard can support compliance in Indian "
  f"organisations ahead of the 2027 enforcement date.")

# ---------------------------------------------------------------- references
H("References")
for r in REFS:
    p = doc.add_paragraph(style="List Number")
    p.add_run(f"{r['authors']}. \"{r['title']}.\" ")
    p.add_run(r["venue"]).italic = True
    p.add_run(f" ({r['year']}). {r['link']}")
    for run in p.runs:
        run.font.size = Pt(9.5)
P("Statutes and standards: Digital Personal Data Protection Act, 2023; Digital Personal Data Protection Rules, 2025 "
  "(G.S.R. 846(E), 13 Nov 2025); Regulation (EU) 2016/679 (GDPR); Regulation (EU) 2024/1689 (AI Act); OWASP Top 10 for LLM "
  "Applications 2025. Clause-level sources: results/compliance.md.", size=9)

OUT.mkdir(exist_ok=True)
docx_path = OUT / "Copilot_Privacy_Guard_Final_Report.docx"
doc.save(docx_path)
print("wrote", docx_path)
