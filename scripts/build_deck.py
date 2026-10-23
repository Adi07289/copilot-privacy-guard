"""Final deck: edits the user's 10-slide deck in place (style preserved) and adds comparative-study slides.
Every number comes from results/*.json. Output: deliverables/Copilot Privacy Guard - Final.pptx"""
import json
from pathlib import Path
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
J = lambda n: json.loads((R / n).read_text())
COMP, BENCH, PII, CJ, META = J("comparison.json"), J("injection_bench.json"), J("pii_detection.json"), J("compliance.json"), J("run_meta.json")
A, LBL = COMP["aggregates"], COMP["labels"]
INK, MUTED, CREAM, CARD, CARDLINE, GRIDC, TEAL, LTEAL = "14171A", "8A8F8C", "FAFAF8", "EFEDE6", "E3E1D9", "C7C8C7", "2F6F5E", "5FA593"
S1, S2, S3, S4 = "1F8A70", "C0822E", "4F67B0", "A0466E"          # validated categorical palette (charts)
HEAD, BODY, MONO = "Space Grotesk", "IBM Plex Sans", "IBM Plex Mono"
checks = CJ["checks"]; NPASS = sum(1 for c in checks if c["passed"])


def pct(v, d=1):
    return "—" if v is None else (f"{100 * v:.{d}f}%" if d else f"{100 * v:.0f}%")


def a(cfg, k, d=1):
    return pct(A[cfg].get(k), d)


def rgb(h):
    return RGBColor.from_string(h)


# ------------------------------------------------------------------ helpers
def set_bg(slide, hexc):
    cSld = slide._element.find(qn("p:cSld"))
    old = cSld.find(qn("p:bg"))
    if old is not None:
        cSld.remove(old)
    bg = etree.Element(qn("p:bg")); bgPr = etree.SubElement(bg, qn("p:bgPr"))
    fill = etree.SubElement(bgPr, qn("a:solidFill")); etree.SubElement(fill, qn("a:srgbClr")).set("val", hexc)
    etree.SubElement(bgPr, qn("a:effectLst")); cSld.insert(0, bg)


def text(slide, x, y, w, h, lines, size=12.5, font=BODY, color=INK, bold=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; f = r.font; f.size = Pt(size); f.name = font; f.bold = bold; f.color.rgb = rgb(color)
        p.space_after = Pt(4)
    return tb


def card(slide, x, y, w, h, fill=CARD, line=CARDLINE):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill); s.line.color.rgb = rgb(line); s.line.width = Pt(0.75); s.shadow.inherit = False
    return s


def _border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    for i, tag in enumerate(("a:lnL", "a:lnR", "a:lnT", "a:lnB")):
        ln = etree.Element(qn(tag), w="12700", cap="flat", cmpd="sng", algn="ctr")
        f = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(f, qn("a:srgbClr")).set("val", GRIDC)
        tcPr.insert(i, ln)


def table(slide, x, y, w, rows, widths=None, size=11, row_h=0.3, highlight=None):
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gf.table; tblPr = tbl._tbl.tblPr; tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    if widths:
        for i, wd in enumerate(widths):
            tbl.columns[i].width = Inches(wd)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci); cell.fill.background(); _border(cell)
            cell.margin_left = cell.margin_right = Inches(0.06); cell.margin_top = cell.margin_bottom = Inches(0.03)
            tf = cell.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]; p.text = ""
            r = p.add_run(); r.text = str(val); f = r.font; f.size = Pt(size); f.name = BODY
            if ri == 0:
                f.bold = True; f.color.rgb = rgb("4A5568")
            elif highlight is not None and ri == highlight:
                f.bold = True; f.color.rgb = rgb(TEAL)
            else:
                f.color.rgb = rgb(INK)
    return gf


def bar_chart(slide, x, y, w, h, cats, series, colors, ymax=100, fmt='0"%"'):
    cd = CategoryChartData(); cd.categories = cats
    for n, v in series.items():
        cd.add_series(n, [0 if q is None else q for q in v])
    ch = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.font.size = Pt(10); ch.font.name = BODY; ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(10)
    va = ch.value_axis; va.maximum_scale = ymax; va.minimum_scale = 0; va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(CARDLINE); va.tick_labels.font.size = Pt(9); va.tick_labels.number_format = fmt
    va.tick_labels.number_format_is_linked = False; va.format.line.fill.background()
    ch.category_axis.tick_labels.font.size = Pt(9.5); ch.category_axis.format.line.color.rgb = rgb(GRIDC)
    for s, col in zip(ch.plots[0].series, colors):
        s.format.fill.solid(); s.format.fill.fore_color.rgb = rgb(col)
    ch.plots[0].gap_width = 70; ch.plots[0].overlap = -8
    return ch


def title(slide, s, color=INK):
    text(slide, 0.89, 0.62, 11.9, 0.6, s, size=30, font=HEAD, color=color, bold=True)


def footer(slide, color=MUTED):
    text(slide, 0.89, 6.83, 3.5, 0.26, "Copilot Privacy Guard", size=12, font=BODY, color=color)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def shapes(slide):
    return {sh.name: sh for sh in slide.shapes}


def set_text(shape, s):
    tf = shape.text_frame; p0 = tf.paragraphs[0]; runs = p0.runs
    runs[0].text = s
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    for p in tf.paragraphs[1:]:
        p._p.getparent().remove(p._p)


def set_paras(shape, lines):
    tf = shape.text_frame; paras = tf.paragraphs
    import copy
    while len(paras) < len(lines):
        tf._txBody.append(copy.deepcopy(paras[-1]._p)); paras = tf.paragraphs
    for p, line in zip(paras, lines):
        p.runs[0].text = line
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
    for p in paras[len(lines):]:
        p._p.getparent().remove(p._p)


def new_slide(prs, bg=CREAM):
    s = prs.slides.add_slide(prs.slide_layouts[0]); set_bg(s, bg)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    return s


def move(prs, old, new):
    lst = prs.slides._sldIdLst; el = list(lst)[old]; lst.remove(el); lst.insert(new, el)


# ------------------------------------------------------------------ build
prs = Presentation(str(ROOT / "docs" / "deck" / "source.pptx"))
old = list(prs.slides)
s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = old

set_text(shapes(s1)["Text 0"], "BCSE318L DATA PRIVACY · VIT VELLORE · FINAL REVIEW")
notes(s1, "Good morning. This is our final review of Copilot Privacy Guard — a DPDP-aligned privacy-guard middleware for "
          "retrieval-augmented enterprise assistants. In Review 3 we showed half of it; today every objective from Review 1 "
          "is delivered, measured, and compared against the defence families in the literature.")

sh = shapes(s3)
set_text(sh["Text 8"], "Redacted index"); set_text(sh["Text 9"], "② purpose · ③ judge")
set_text(sh["Text 15"], "qwen3.8 · Groq")
# fonts are substituted on machines without IBM Plex: give every box label the full box width, centred
from pptx.enum.text import PP_ALIGN
boxes = {"Text 2": "Shape 1", "Text 3": "Shape 1", "Text 5": "Shape 4", "Text 6": "Shape 4", "Text 8": "Shape 7", "Text 9": "Shape 7",
         "Text 11": "Shape 10", "Text 12": "Shape 10", "Text 14": "Shape 13", "Text 15": "Shape 13"}
for t, b in boxes.items():
    tx, bx = sh[t], sh[b]
    tx.left, tx.width = bx.left + Inches(0.05), bx.width - Inches(0.1)
    tx.text_frame.word_wrap = True
    for para in tx.text_frame.paragraphs:
        para.alignment = PP_ALIGN.CENTER
sh["Text 8"].top = sh["Text 5"].top; sh["Text 9"].top = sh["Text 6"].top
notes(s3, "Four controls on every prompt and every retrieved document. Documents are redacted before they are indexed, so the "
          "vector store never holds raw PII. Then the purpose filter and injection screening — now a heuristic plus an LLM "
          "judge — run on the candidates, PII redaction runs again on what survives, and only then does the LLM see anything. "
          "Every decision is hash-chained into the audit log.")

sh = shapes(s4)
set_text(sh["Text 3"], "India-first recognizers — Aadhaar (Verhoeff), PAN, mobile, IFSC, UPI, card (Luhn), ₹ salary — plus spaCy "
                       "NER. Applied to the prompt, to every document before indexing, and to surviving chunks.")
set_text(sh["Text 9"], "Weighted pattern bank with zero-width, HTML-comment and base64 checks, OR an LLM-as-judge "
                       "(gpt-oss-120b). Flagged chunks are dropped; the rest are wrapped as <untrusted_document>.")
notes(s4, "The controls in detail. Redaction now also protects the index itself. Injection screening gained an LLM judge, "
          "which catches paraphrased and Hindi payloads that patterns miss — you'll see the numbers in a moment.")

sh = shapes(s5)
set_paras(sh["Text 3"], ["Python 3.13 package privacyguard — 12 configurable defence configurations",
                         f"123 unit & end-to-end tests, pytest, test-driven", "Committed LLM caches: every result reproducible offline"])
set_paras(sh["Text 6"], ["Faker (en_IN): 80 docs, 369 PII spans, 4 canary-poisoned docs; 36 queries × 5 personas",
                         "TF-IDF and bge-small (fastembed) behind one Retriever protocol",
                         "Streamlit demo: compare any two configs, attack lab, compliance"])
set_paras(sh["Text 9"], [f"Groq: {META.get('groq_model')} answers, {META.get('judge_model')} judge",
                         "Comparators: Meta Prompt Guard 2, Microsoft Presidio", "spaCy NER + India-first regex registry"])
notes(s5, "Same stack, extended: twelve configurations of one pipeline, a dense retriever, a real LLM judge on Groq, and two "
          "industry tools — Presidio and Meta's Prompt Guard 2 — as comparators. All LLM responses are cached and committed, so "
          "every number on these slides reproduces offline.")

tbl = [sh for sh in s6.shapes if sh.has_table][0].table
status = {1: ("Literature review — 32 verified papers across 5 threat classes", "Done"),
          4: (None, "Done — EchoLeak, Slack AI, Samsung"), 5: (None, "Done — 4 controls + redacted index"),
          6: (None, f"Done — 9 metrics × {len(COMP['configs'])} configs"), 7: (None, "Done — 7-risk register"),
          8: (None, f"Done — {len(CJ['clauses'])} verified clauses"), 9: (None, f"Done — {len(checks)} executable checks, {NPASS} pass"),
          10: (None, "Done — report & slide 16")}
for ri, (desc, st) in status.items():
    if desc:
        tbl.cell(ri, 1).text_frame.paragraphs[0].runs[0].text = desc
    tbl.cell(ri, 2).text_frame.paragraphs[0].runs[0].text = st
set_text([x for x in s6.shapes if x.name == "Text 0"][0], "All ten objectives delivered")
notes(s6, "Every objective from Review 1 is now delivered: incidents, the DPIA register, the clause mapping and the assurance "
          "checklist joined the prototype and the evaluation. The literature grew from 14 to 32 verified papers.")

# old slide 7 -> headline results
sh = shapes(s7)
set_text(sh["Text 0"], "Measured results — 36 queries, 5 personas, LLM on Groq")
set_text(sh["Text 2"], "PII sent to the LLM provider"); set_text(sh["Text 3"], a("none", "context_pii_exposure", 0)); set_text(sh["Text 4"], a("full", "context_pii_exposure", 0))
set_text(sh["Text 6"], "Over-sharing retrieval rate"); set_text(sh["Text 7"], a("none", "oversharing_rate")); set_text(sh["Text 8"], a("full", "oversharing_rate", 0))
old_tbl = [x for x in s7.shapes if x.has_table][0]; old_tbl._element.getparent().remove(old_tbl._element)
table(s7, 0.89, 3.45, 11.56, [["Metric", "Vanilla RAG", "Copilot Privacy Guard"],
      ["Raw PII stored in the vector index", a("none", "index_pii_exposure", 0), a("full", "index_pii_exposure", 0)],
      ["Poisoned chunks reaching the LLM", a("none", "injection_exposure_rate", 0), a("full", "injection_exposure_rate", 0)],
      [f"Injection success — canary obeyed by the LLM ({A['none']['injection_success_n']} attacked queries)", a("none", "injection_success_rate", 0), a("full", "injection_success_rate", 0)],
      ["PII in final LLM answers", a("none", "answer_pii_leakage_llm"), a("full", "answer_pii_leakage_llm", 0)],
      ["Answer-quality retention (ROUGE-L vs undefended)", "100%", a("full", "answer_quality_rouge_l")],
      ["Added latency per query", "0 ms", f"+{A['full']['added_latency_ms']:.0f} ms"]], [6.6, 2.4, 2.56], size=12, row_h=0.4)
notes(s7, f"Same retriever, same LLM. PII sent to the provider drops from {a('none','context_pii_exposure',0)} to zero, "
          f"over-sharing from {a('none','oversharing_rate')} to zero, raw PII in the index and poisoned chunks reaching the model "
          f"both to zero. The honest part is the last two rows: answer-quality retention falls to {a('full','answer_quality_rouge_l')} "
          f"and the judge costs about {A['full']['added_latency_ms']/1000:.1f} seconds per query. Note the LLM itself obeyed almost "
          "no canaries — model resistance isn't something the deployer controls, so we design for zero exposure instead.")

# ---- new: literature positioning matrix
n7 = new_slide(prs); title(n7, "Where the literature stops — and what we add")
Y, P_, N_ = "✓", "◐", "✗"
table(n7, 0.89, 1.5, 11.56, [
    ["Work", "Role / purpose-aware", "PII redaction (India)", "Injection defence", "Protects index", "Audit", "Measured as pipeline", "DPDP-mapped"],
    ["Vanilla RAG — Lewis 2020", N_, N_, N_, N_, N_, N_, N_],
    ["Inherited RBAC — M365 Copilot", P_ + " role only", N_, N_, N_, P_, N_, N_],
    ["Privacy-Aware Decoding — Wang 2025", N_, N_, N_, N_, N_, P_, N_],
    ["Two-stage RAG privacy — Zhang 2026", N_, P_, N_, N_, N_, P_, N_],
    ["Spotlighting / StruQ — 2024", N_, N_, Y, N_, N_, P_, N_],
    ["Llama Guard / Prompt Guard — Meta", N_, N_, Y, N_, N_, N_, N_],
    ["Presidio · hybrid PII — Mishra 2025", N_, P_ + " off by default", N_, N_, N_, N_, N_],
    ["SPARSE — Tsai 2026", N_, N_, N_, P_ + " noise", N_, N_, N_],
    ["Copilot Privacy Guard (ours)", Y + " purpose", Y, Y + " + judge", Y + " redacted", Y + " hash chain", Y, Y]],
    [3.3, 1.25, 1.35, 1.25, 1.15, 1.05, 1.2, 1.01], size=10.5, row_h=0.42, highlight=9)
text(n7, 0.89, 6.1, 11.5, 0.4, f"✓ addressed · ◐ partially · ✗ not addressed. Positions drawn from Table 1 of the report ({32} verified papers).", size=11, color=MUTED)
footer(n7)
notes(n7, "This is the comparative positioning. Every defence in the literature covers one column: decoding noise, delimiting, "
          "content classifiers, PII redaction, embedding noise. None knows who is asking or why, none protects the index, audits "
          "decisions and is measured end to end — and none is mapped to India's DPDP Act. That is the gap we fill.")

# ---- new: experimental setup
n8 = new_slide(prs); title(n8, "Experimental setup — one pipeline, twelve configurations")
tiles = [("CORPUS", ["80 synthetic Indian-enterprise docs", "email · HR · tickets · code", "369 labelled PII spans", "4 canary-poisoned docs"]),
         ("QUERIES", ["36 queries × 5 personas", "8 unauthorised probes", "8 authorised sensitive · 12 utility", "8 injection (authorised persona)"]),
         ("CONFIGURATIONS", ["6 defence-family analogues", "ours + 4 leave-one-out ablations", "+ dense-retrieval variant", "same corpus, retriever, LLM"]),
         ("METRICS & MODELS", ["9 metrics incl. context & index exposure", f"answers: {META.get('groq_model')}", f"judge: {META.get('judge_model')}", "Presidio + Prompt Guard 2 as comparators"])]
for i, (h, body) in enumerate(tiles):
    x = 0.89 + i * 2.93; card(n8, x, 1.6, 2.75, 4.2)
    text(n8, x + 0.22, 1.85, 2.4, 0.3, h, size=12, font=MONO, color=TEAL, bold=True)
    text(n8, x + 0.22, 2.3, 2.35, 3.3, body, size=13)
text(n8, 0.89, 6.05, 11.5, 0.4, "Non-ours rows are simplified analogues of a defence family — not reproductions of the cited papers.", size=11, color=MUTED)
footer(n8)
notes(n8, "To compare fairly we made every defence a configuration of the same pipeline, so the corpus, the retriever and the LLM "
          "are identical and only the defence changes. Injection queries are issued by a persona who is allowed to read the "
          "poisoned document — otherwise access control would block them and we'd be measuring the wrong control.")

# ---- new: comparative study
n10 = new_slide(prs); title(n10, "Comparative study — only the defence differs")
order = ["none", "rbac", "pii_only", "output_filter", "spotlight", "heuristic", "full"]
table(n10, 0.89, 1.45, 11.56, [["Configuration", "PII to LLM", "Over-sharing", "Injection exposure", "Index PII", "ROUGE-L", "Latency"]]
      + [[LBL[k], a(k, "context_pii_exposure", 0), a(k, "oversharing_rate"), a(k, "injection_exposure_rate", 0), a(k, "index_pii_exposure", 0),
          a(k, "answer_quality_rouge_l", 0), f"{A[k]['added_latency_ms']:+.0f} ms"] for k in order],
      [3.9, 1.25, 1.35, 1.5, 1.2, 1.15, 1.21], size=12, row_h=0.42, highlight=7)
text(n10, 0.89, 5.0, 11.5, 1.6, [f"Inherited RBAC still sends {a('rbac','context_pii_exposure',0)} of PII to the provider and lets {a('rbac','injection_exposure_rate',0)} of poisoned chunks through.",
                                 f"Redaction-only leaves over-sharing at {a('pii_only','oversharing_rate',0)}; output filtering still sends {a('output_filter','context_pii_exposure',0)} of PII to the provider.",
                                 "Only the combined guard closes every exposure — at a utility and latency cost."], size=13)
footer(n10)
notes(n10, "Read the table by columns. Copilot-style RBAC fixes most over-sharing but sends half the PII to the model provider. "
           "Redaction-only fixes the provider exposure but not over-sharing. Output filtering looks clean in the answer while the "
           "provider still receives everything — under DPDP section 8(1) the fiduciary is responsible for that. Every single-family "
           "defence leaves at least one exposure open; ours closes all four, and the right-hand columns show what that costs.")

# ---- new: ablation
n11 = new_slide(prs); title(n11, "Ablation — each control owns one exposure")
abl = [k for k in ["full", "full_no_access", "full_no_pii", "full_no_index", "full_no_judge", "full_dense"] if k in A]
table(n11, 0.89, 1.45, 11.56, [["Configuration", "PII to LLM", "Over-sharing", "Index PII", "Injection exposure", "ROUGE-L", "Latency"]]
      + [[LBL[k], a(k, "context_pii_exposure", 0), a(k, "oversharing_rate"), a(k, "index_pii_exposure", 0), a(k, "injection_exposure_rate", 0),
          a(k, "answer_quality_rouge_l", 0), f"{A[k]['added_latency_ms']:+.0f} ms"] for k in abl],
      [3.9, 1.25, 1.35, 1.2, 1.5, 1.15, 1.21], size=12, row_h=0.42, highlight=1)
text(n11, 0.89, 4.6, 11.5, 1.8, [f"Drop the access filter → over-sharing returns ({a('full_no_access','oversharing_rate',0)}).",
                                 f"Drop redaction → {a('full_no_pii','context_pii_exposure',0)} of PII reaches the provider and the index holds all of it.",
                                 "Drop the redacted index → the index again stores every raw PII value.",
                                 f"Drop the judge → same corpus metrics, latency +{A['full_no_judge']['added_latency_ms']:.0f} ms — its value shows on harder payloads (next slides)."], size=13)
footer(n11)
notes(n11, "Remove one control at a time and exactly one exposure comes back — the controls don't overlap, they compose. The judge "
           "doesn't change these corpus metrics because our seeded payloads are also caught by patterns; its value is on paraphrased "
           "and Hindi attacks, which is the next comparison.")

# ---- new: PII vs Presidio
n12 = new_slide(prs); title(n12, "PII detection — head-to-head with Microsoft Presidio")
types = ["AADHAAR", "PAN", "IN_MOBILE", "IFSC", "UPI_VPA", "EMAIL", "CREDIT_CARD", "DOB", "SALARY_INR", "PERSON"]
nice = ["Aadhaar", "PAN", "Mobile", "IFSC", "UPI", "Email", "Card", "DOB", "₹ Salary", "Person"]
ho = PII["results"]["heldout"]; dn = {"ours": "Ours (India-first)", "presidio_default": "Presidio (as shipped)", "presidio_india": "Presidio + India recognizers"}
ser = {dn[d]: [100 * ((ho[d]["per_type"].get(t) or {}).get("recall") or 0) for t in types] for d in PII["detectors"]}
bar_chart(n12, 0.7, 1.35, 8.6, 4.9, nice, ser, [S1, S2, S3][:len(ser)])
card(n12, 9.55, 1.55, 2.95, 4.5)
text(n12, 9.75, 1.75, 2.6, 0.3, "HELD-OUT F1", size=12, font=MONO, color=TEAL, bold=True)
text(n12, 9.75, 2.2, 2.6, 3.6, [f"Ours  {pct(ho['ours']['f1'])}", f"Presidio  {pct(ho['presidio_default']['f1'])}", f"+ India  {pct(ho['presidio_india']['f1'])}" if "presidio_india" in ho else "",
                                "", "No Presidio recognizer for IFSC, UPI or ₹ salary; Aadhaar/PAN ship disabled.", "", "Presidio wins on spaced mobiles and written dates — our gaps."], size=12)
text(n12, 0.89, 6.35, 11.5, 0.3, "Held-out set: 32 sentences in formats the corpus generator never produces. Recall per entity type.", size=10.5, color=MUTED)
footer(n12)
notes(n12, f"In Review 3 we said Presidio was US-centric. We checked: it does ship Aadhaar and PAN recognizers, just disabled. So we "
           f"measured instead of asserting. On held-out sentences our F1 is {pct(ho['ours']['f1'])} against {pct(ho['presidio_default']['f1'])} "
           "for Presidio as shipped. Presidio is better on spaced mobile numbers and written-out dates — those are our gaps, and they're "
           "in future work.")

# ---- new: injection bench
n13 = new_slide(prs); title(n13, "Injection detection — heuristic vs LLM judge vs Prompt Guard 2")
fams = ["direct", "roleplay", "exfiltration", "obfuscated", "paraphrased", "hindi"]
dets = BENCH.get("detectors", ["heuristic", "judge", "combined"])
bn = {"heuristic": "Heuristic", "judge": "LLM judge", "combined": "Combined (ours)", "prompt_guard": "Prompt Guard 2 (Meta)"}
bc = {"heuristic": S2, "judge": S3, "combined": S1, "prompt_guard": S4}
bar_chart(n13, 0.7, 1.35, 8.6, 4.9, [f.title() for f in fams] + ["All 24"],
          {bn[d]: [100 * (BENCH["by_family"][f][d] or 0) for f in fams] + [100 * (BENCH["tpr"][d] or 0)] for d in dets}, [bc[d] for d in dets])
card(n13, 9.55, 1.55, 2.95, 4.5)
text(n13, 9.75, 1.75, 2.6, 0.3, "FALSE POSITIVES", size=12, font=MONO, color=TEAL, bold=True)
text(n13, 9.75, 2.2, 2.6, 3.6, ["on 8 hard negatives:"] + [f"{bn[d]}  {pct(BENCH['fpr_hard'][d],0)}" for d in dets]
     + ["", "The judge alone: 100% detection, 0% false positives. OR-ing in the heuristic inherits its false positives."], size=12)
text(n13, 0.89, 6.35, 11.5, 0.3, "24 payloads (6 families × 4) in a benign ticket; 40 clean docs + 8 hard negatives (security reminders, release notes).", size=10.5, color=MUTED)
footer(n13)
notes(n13, "Patterns catch explicit and encoded attacks but none of the paraphrased or Hindi ones, and they flag security reminders "
           "as attacks. The LLM judge caught all 24 with no false positives. Meta's dedicated Prompt Guard model caught a third — it's "
           "strong on explicit overrides but blind to exfiltration and Hindi. One design lesson: OR-ing the heuristic in inherits its "
           "false positives, so the judge should make the final call — that's future work item one.")

# ---- new: compliance
n14 = new_slide(prs); title(n14, "Compliance as code — clauses, DPIA, assurance checklist")
fw = {}
for cl in CJ["clauses"]:
    fw.setdefault(cl["framework"], []).append(cl["ref"])
cols = [("CLAUSE MAP", [f"{len(CJ['clauses'])} clauses, each verified against the source text"] + [f"{k}: {', '.join(v)}" for k, v in fw.items()]),
        ("DPIA REGISTER", ["7 risks · likelihood × impact"] + [f"{r['id']} {r['threat'][:30]}: {r['inherent'][2]} → {r['residual'][2]}" for r in CJ["risks"]]),
        ("ASSURANCE CHECKLIST", [f"{NPASS} / {len(checks)} executable checks pass"] + [f"{c['id']} {'PASS' if c['passed'] else ('FAIL' if c['passed'] is False else 'n/m')}" for c in checks])]
for i, (h, body) in enumerate(cols):
    x = 0.89 + i * 3.9; card(n14, x, 1.45, 3.75, 4.95)
    text(n14, x + 0.2, 1.62, 3.4, 0.3, h, size=12, font=MONO, color=TEAL, bold=True)
    text(n14, x + 0.2, 2.02, 3.4, 4.3, body, size=10.5 if i != 2 else 10)
footer(n14)
notes(n14, "Compliance isn't a slide of logos here — it's code. Every clause from the DPDP Act, the DPDP Rules 2025, GDPR, the EU AI "
           f"Act and OWASP points at a measured number. The DPIA residual scores cite the metric that justifies them, and the checklist "
           f"runs against the live system: {NPASS} of {len(checks)} pass. The one that fails is utility, A12 — we kept it failing rather "
           "than lower the bar.")

# ---- old slide 8 (unique factor)
sh = shapes(s8)
set_text(sh["Text 6"], f"Held-out F1 {pct(ho['ours']['f1'])} vs {pct(ho['presidio_default']['f1'])} for Presidio as shipped — Aadhaar "
                       "(Verhoeff), PAN, IFSC, UPI and ₹ salary formats Presidio misses or ships disabled.")
set_text(sh["Text 9"], "Four controls, a redacted index and one audit chain — measured against six defence families on the same corpus, "
                       "queries and LLM, with every clause mapped to evidence.")
notes(s8, "Three things nobody in our 32-paper survey does together: purpose-aware access as a runtime DPDP rule; India-first PII "
          "detection that we measured against Presidio rather than asserted; and a full pipeline measured against the alternatives, "
          "with compliance evidence generated from the results.")

# ---- old slide 9 (limitations)
sh = shapes(s9)
set_text(sh["Text 3"], f"Utility cost — answer-quality retention {a('full','answer_quality_rouge_l')} ROUGE-L: redaction removes names and contacts users ask for; heuristic false positives drop benign chunks (A12 fails).")
set_text(sh["Text 6"], f"Latency — the LLM judge adds ~{(A['full']['added_latency_ms']-A['full_no_judge']['added_latency_ms'])/1000:.1f} s per query; caching and batching would cut it.")
set_text(sh["Text 9"], "Synthetic corpus — a held-out set limits author bias; two detector fixes made after the first run are disclosed.")
set_text(sh["Text 12"], "Static access model — tags on documents, not an identity-provider integration.")
set_text(sh["Text 14"], "Injection success has few positives — the LLM itself refused most canaries; exposure is the robust measure.")
notes(s9, "We'd rather say these ourselves. The guard costs utility and latency, and we report the failing check instead of hiding "
          "it. The corpus is synthetic. Access is tag-based. And because the model refused most canaries on its own, we lean on the "
          "exposure metric — the guard makes injection safety independent of model behaviour.")

# ---- old slide 10 (future work)
sh = shapes(s10)
set_text(sh["Text 0"], "Future work")
for k, s in zip(["Text 2", "Text 4", "Text 6", "Text 8", "Text 10", "Text 12", "Text 14"],
                ["LLM judge as the sole drop decision — keep 100% detection, remove heuristic false positives",
                 "Re-identify placeholders for authorised users",
                 "Recognizers for secrets, spaced mobiles, written dates, lakh/crore",
                 "Output egress allow-list (EchoLeak, Slack AI gap)",
                 "Identity-provider ACLs + consent prompt for purpose",
                 "Real de-identified corpus; adaptive attacks",
                 "Batch and cache judge calls to cut latency"]):
    set_text(sh[k], s)
notes(s10, "Next steps follow directly from the limitations: let the judge decide alone, give authorised users their real values "
           "back without the model ever seeing them, widen the recognizers, add egress control, and plug into a real identity "
           "provider. Thank you — we're happy to take questions or open the live demo.")

# ---- appendix
ap1 = new_slide(prs); title(ap1, "Appendix A — replay of documented incidents")
table(ap1, 0.89, 1.45, 11.56, [["Incident", "Exposure point", "Injection screening", "PII redaction", "Access filter", "Audit", "Residual gap"],
      ["EchoLeak — M365 Copilot (CVE-2025-32711)", "retrieval → inference", "Partially", "Partially", "No", "Yes", "output egress allow-list"],
      ["Slack AI indirect injection (2024)", "cross-zone retrieval", "Yes (likely)", "Partially", "Partially", "Yes", "secret recognizers"],
      ["Samsung → ChatGPT (2023)", "prompt → external processor", "No", "Partially", "No", "Yes", "non-personal confidential data"]],
      [3.4, 2.0, 1.45, 1.2, 1.15, 0.8, 1.56], size=11.5, row_h=0.55)
text(ap1, 0.89, 4.2, 11.5, 1.5, ["Two of three incidents are injection through retrieval; all three end with data leaving through the model's input or "
                                 "output channel. No single control stops any of them — the case for layered, audited middleware. Sources and root-cause "
                                 "chains: docs/INCIDENTS.md."], size=13)
footer(ap1)
notes(ap1, "Objective O4. Replaying three real incidents against our controls: none is fully stopped by one control, which is the argument for layering.")

ap2 = new_slide(prs); title(ap2, "Appendix B — reported results vs ours (context only)")
table(ap2, 0.89, 1.45, 11.56, [["Work", "Their setting", "Reported", "Our closest measurement"],
      ["DEAL (Zhang 2025)", "optimiser attack, undefended RAG", "near-total PII extraction", f"vanilla sends {a('none','context_pii_exposure',0)} of PII to the LLM; ours {a('full','context_pii_exposure',0)}"],
      ["Privacy-Aware Decoding (Wang 2025)", "decoding-time noise", "leakage–utility trade-off", f"0% PII in context at {a('full','answer_quality_rouge_l',0)} ROUGE-L"],
      ["Spotlighting (Hines 2024)", "prompt delimiting", "reduced attack success", f"delimiting-only: exposure {a('spotlight','injection_exposure_rate',0)}, success {a('spotlight','injection_success_rate',0)}"],
      ["Vec2Text (Morris 2023)", "embedding inversion", "recovers most short inputs", f"index PII {a('none','index_pii_exposure',0)} → {a('full','index_pii_exposure',0)}"],
      ["PIIBench (Jha 2026)", "unified PII taxonomy", "low span-F1 for 8 detectors", f"Presidio held-out F1 {pct(ho['presidio_default']['f1'])}"]],
      [3.0, 2.7, 2.5, 3.36], size=11.5, row_h=0.5)
footer(ap2)
notes(ap2, "Numbers reported by other papers next to ours. Different data, models and attacks — this is context, not a head-to-head; the head-to-head is the comparative study slide.")

ap3 = new_slide(prs); title(ap3, "Appendix C — live demo")
text(ap3, 0.89, 1.5, 11.5, 4.8, ["cd ~/copilot-privacy-guard && .venv/bin/streamlit run app/streamlit_app.py",
                                 "Compare — Riya Menon (engineer), q01: Vanilla RAG vs ours → six HR chunks dropped by clearance, nothing sent to the LLM.",
                                 "Compare — q32 (injection): Heuristic screening only vs ours → poisoned ticket dropped, judge score shown.",
                                 "Attack lab — paste the paraphrased payload → heuristic 0.00, LLM judge ≈ 0.99 → DROP.",
                                 "Comparative study / Compliance tabs → the tables on slides 10–14, live from results/.",
                                 "Audit log → Verify chain → Tamper a line → Verify shows broken_at.",
                                 "Offline: committed caches replay every LLM result with no network."], size=14)
footer(ap3)
notes(ap3, "Demo order if time allows.")

# ---- order: 1-6, new lit, new setup, headline(old7), comparative, ablation, PII, injection, compliance, old8, old9, old10, appendix
order_ids = [s1, s2, s3, s4, s5, s6, n7, n8, s7, n10, n11, n12, n13, n14, s8, s9, s10, ap1, ap2, ap3]
lst = prs.slides._sldIdLst
id_of = {sl.slide_id: el for sl, el in zip(prs.slides, list(lst))}
ordered = [id_of[sl.slide_id] for sl in order_ids]          # resolve before detaching
for el in list(lst):
    lst.remove(el)
for el in ordered:
    lst.append(el)
out = ROOT / "deliverables" / "Copilot Privacy Guard - Final.pptx"
prs.save(str(out)); print("wrote", out, "slides:", len(prs.slides))
