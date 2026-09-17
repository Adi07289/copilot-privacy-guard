"""Side-by-side demo: Baseline RAG vs Guarded RAG on the synthetic enterprise corpus."""
from __future__ import annotations
import html, os, re
from pathlib import Path
import streamlit as st
from privacyguard.controls.audit import HashChainedAuditLog
from privacyguard.corpus.generator import PERSONAS, generate_corpus, load_corpus, save_corpus
from privacyguard.eval.harness import run_eval, save_report
from privacyguard.eval.queries import QUERY_SET
from privacyguard.llm import ExtractiveProvider, get_provider
from privacyguard.models import PURPOSES, SENSITIVITY_LABELS
from privacyguard.pipeline import build_default

CORPUS_PATH = Path(os.getenv("PG_CORPUS_PATH", "data/corpus.json"))
AUDIT_PATH = Path(os.getenv("PG_AUDIT_PATH", "data/audit.jsonl"))
RESULTS_MD = Path("results/eval.md")
_DEFAULT_PURPOSE = {"HR": "hr_operations", "Engineering": "engineering", "Finance": "finance_reporting",
                    "Legal": "legal_review", "Sales": "sales_support"}

st.set_page_config(page_title="Privacy-Guard Middleware", layout="wide")


@st.cache_resource
def _load():
    if not CORPUS_PATH.exists():
        save_corpus(generate_corpus(), CORPUS_PATH)
    corpus = load_corpus(CORPUS_PATH)
    base, guard = build_default(corpus, audit_path=AUDIT_PATH)
    return corpus, base, guard


corpus, base, guard = _load()
by_id = {d.id: d for d in corpus}
gold_values = sorted({s.value for d in corpus for s in d.pii_spans}, key=len, reverse=True)


def _highlight(text: str, guarded: bool) -> str:
    out = html.escape(text)
    if guarded:
        out = re.sub(r"&lt;([A-Z_]+_\d+)&gt;", r'<mark style="background:#c8f7c5">&lt;\1&gt;</mark>', out)
    else:
        for v in gold_values:
            out = out.replace(html.escape(v), f'<mark style="background:#ffb3b3">{html.escape(v)}</mark>')
    return f'<div style="white-space:pre-wrap;font-family:monospace;font-size:12px">{out}</div>'


with st.sidebar:
    st.title("Privacy-Guard")
    st.caption("Baseline RAG vs. four-control guarded middleware")
    persona = st.selectbox("Persona", PERSONAS, format_func=lambda u: f"{u.name} — {u.role} ({u.department}, clearance {u.clearance})")
    purpose = st.selectbox("Declared purpose", PURPOSES, index=PURPOSES.index(_DEFAULT_PURPOSE[persona.department]))
    canned = st.selectbox("Canned query", QUERY_SET, format_func=lambda q: f"{q.id}: {q.text[:60]}")
    free = st.text_area("…or type your own", "")
    query = free.strip() or canned.text
    st.caption(f"LLM provider: `{getattr(base.provider, 'name', 'extractive')}` · names: `{guard.pii.detector.names_mode}`")
    run = st.button("Run query", key="run", type="primary")

tab_cmp, tab_audit, tab_eval = st.tabs(["Compare", "Audit log", "Evaluation"])

with tab_cmp:
    st.markdown(f"**Query:** {query}")
    if run:
        st.session_state["last"] = (base.run(persona, purpose, query), guard.run(persona, purpose, query))
    if "last" in st.session_state:
        b, g = st.session_state["last"]
        c1, c2 = st.columns(2)
        for col, res, title, guarded in ((c1, b, "Baseline RAG", False), (c2, g, "Guarded RAG", True)):
            with col:
                st.subheader(title)
                st.caption(f"{res.latency_ms:.0f} ms · provider `{res.provider_used}` · {len(res.context_sent)} chunks sent to LLM")
                if guarded:
                    st.markdown(f"**Redacted query sent to retriever/LLM:** `{res.redacted_query}`")
                for ch in res.chunks:
                    d = by_id[ch.doc_id]
                    badges = " ".join(f"`{f}`" for f in ch.flags) + (f" `dropped: {ch.drop_reason}`" if ch.dropped else "")
                    with st.expander(f"{ch.doc_id} · {d.department} · {SENSITIVITY_LABELS[d.sensitivity]} · score {ch.score:.2f} {badges}"):
                        st.markdown(_highlight(ch.text, guarded), unsafe_allow_html=True)
                st.markdown("**Answer**")
                st.markdown(_highlight(res.answer, guarded), unsafe_allow_html=True)
        st.markdown("**Guarded decisions (what each control did)**")
        st.dataframe([{"control": d.control, "action": d.action, "doc": d.doc_id, **{k: str(v) for k, v in d.detail.items()}} for d in g.decisions], use_container_width=True)

with tab_audit:
    log = HashChainedAuditLog(AUDIT_PATH)
    a1, a2, a3 = st.columns(3)
    if a1.button("Verify chain", key="verify"):
        v = log.verify()
        (st.success if v.ok else st.error)(f"ok={v.ok} · entries={v.total} · broken_at={v.broken_at}")
    if a2.button("Tamper a line", key="tamper"):
        v = log.verify()
        if v.total:
            seq = max(1, v.total // 2)
            log.tamper(seq); st.warning(f"mutated entry {seq} on disk — now click Verify chain")
    if a3.button("Reset log", key="reset"):
        log.clear(); st.info("cleared")
    st.dataframe([{"seq": e["seq"], "control": e["event"].get("control"), "action": e["event"].get("action"), "doc": e["event"].get("doc_id"), "hash": e["hash"][:16] + "…", "prev": e["prev_hash"][:16] + "…"} for e in log.tail(20)], use_container_width=True)

with tab_eval:
    if st.button("Run evaluation (extractive provider, ~5 s)", key="eval"):
        rep = run_eval(corpus, QUERY_SET, ExtractiveProvider(), AUDIT_PATH)
        save_report(rep, "results/eval.json", RESULTS_MD)
    if RESULTS_MD.exists():
        st.markdown(RESULTS_MD.read_text())
    else:
        st.info("No results yet — click the button above or run `scripts/run_eval.py`.")
