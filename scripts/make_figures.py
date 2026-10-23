"""Figures for the report and deck. Every number is read from results/*.json — none is typed in.
Palette validated with the dataviz validator (lightness, chroma, CVD, normal-vision, contrast: all pass)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "docs" / "figures"
TEAL, AMBER, BLUE, PLUM = "#1F8A70", "#C0822E", "#4F67B0", "#A0466E"   # categorical slots 1-4
GREY, INK, MUTED, GRID = "#B9BDBA", "#14171A", "#5B605D", "#E6E7E5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def load(name):
    return json.loads((R / name).read_text())


def _pct(v):
    return None if v is None else 100 * v


def _hbar_panel(ax, labels, values, title, highlight, unit="%", xmax=100):
    ys = range(len(labels))[::-1]
    for y, lab, v in zip(ys, labels, values):
        if v is None:
            ax.text(1, y, "not measured", va="center", fontsize=7, color=MUTED)
            continue
        ax.barh(y, v, height=0.62, color=TEAL if lab in highlight else GREY, edgecolor="white", linewidth=1)
        ours = lab in highlight
        ax.text(v + xmax * 0.015, y, f"{v:.0f}{unit}" if unit == "%" else f"{v:+.0f}{unit}", va="center", fontsize=7.5 if ours else 7,
                color=TEAL if ours else INK, fontweight="bold" if ours else "normal")
    ax.set_yticks(list(ys)); ax.set_yticklabels(labels)
    for t in ax.get_yticklabels():
        if t.get_text() in highlight:
            t.set_color(TEAL); t.set_fontweight("bold")
    ax.set_xlim(0, xmax * 1.18); ax.set_title(title, fontsize=9, color=INK, loc="left", pad=6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", labelsize=7)


def fig_comparison(comp, keys, rows, fname, title):
    agg, labels = comp["aggregates"], comp["labels"]
    rows = [c for c in rows if c in agg]
    names = [labels[c] for c in rows]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.55 * len(keys), 0.34 * len(rows) + 1.0), sharey=True)
    for ax, (k, t) in zip(axes, keys):
        _hbar_panel(ax, names, [_pct(agg[c].get(k)) for c in rows], t, {labels["full"]})
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK, y=1.02)
    fig.savefig(OUT / fname); plt.close(fig)


def fig_privacy_utility(comp):
    agg, labels = comp["aggregates"], comp["labels"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for c in comp["configs"]:
        x, y = _pct(agg[c]["context_pii_exposure"]), _pct(agg[c].get("answer_quality_rouge_l"))
        if x is None or y is None:
            continue
        ours = c == "full"
        ax.scatter(x, y, s=70 if ours else 40, color=TEAL if ours else GREY, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(labels[c], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7,
                    color=INK if ours else MUTED, fontweight="bold" if ours else "normal")
    ax.set_xlabel("Context PII exposure to the LLM provider (%) — lower is better")
    ax.set_ylabel("Answer-quality retention, ROUGE-L (%) — higher is better")
    ax.set_xlim(-5, 110); ax.set_ylim(0, 108); ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_title("Privacy–utility trade-off across defence configurations", loc="left", fontsize=10)
    fig.savefig(OUT / "fig_privacy_utility.png"); plt.close(fig)


def fig_grouped(cats, series, colors, fname, title, ylabel, ymax=100):
    import numpy as np
    x = np.arange(len(cats)); n = len(series); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(6.4, 0.62 * len(cats) * n / 2 + 2), 3.6))
    for i, ((name, vals), col) in enumerate(zip(series.items(), colors)):
        xs = x - 0.4 + w / 2 + i * w
        ax.bar(xs, [0 if v is None else v for v in vals], width=w * 0.92, color=col, label=name, edgecolor="white", linewidth=1)
        for xi, v in zip(xs, vals):
            if v is None:
                ax.text(xi, 1.5, "n/a", ha="center", fontsize=6, color=MUTED, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8); ax.set_ylim(0, ymax * 1.08); ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, ncol=n, loc="upper left", bbox_to_anchor=(0, 1.13))
    ax.set_title(title, loc="left", fontsize=10, pad=24)
    fig.savefig(OUT / fname); plt.close(fig)


def fig_pii(pii):
    res = pii["results"]["heldout"]
    order = ["AADHAAR", "PAN", "IN_MOBILE", "IFSC", "UPI_VPA", "EMAIL", "CREDIT_CARD", "DOB", "SALARY_INR", "PERSON"]
    names = {"ours": "Ours (India-first)", "presidio_default": "Presidio (as shipped)", "presidio_india": "Presidio + India recognizers"}
    series = {names[d]: [_pct((res[d]["per_type"].get(t) or {}).get("recall")) if t in res[d]["per_type"] else 0.0 for t in order]
              for d in pii["detectors"]}
    fig_grouped([t.replace("_", " ").title().replace("Upi Vpa", "UPI").replace("In Mobile", "Mobile").replace("Ifsc", "IFSC")
                 .replace("Pan", "PAN").replace("Dob", "DOB").replace("Salary Inr", "Salary ₹") for t in order],
                series, [TEAL, AMBER, BLUE], "fig_pii_vs_presidio.png",
                "PII recall per entity type on the held-out set (formats the generator never produces)", "Recall (%)")


def fig_bench(bench):
    fams = ["direct", "roleplay", "exfiltration", "obfuscated", "paraphrased", "hindi"]
    dets = bench.get("detectors", ["heuristic", "judge", "combined"])
    names = {"heuristic": "Heuristic", "judge": "LLM judge", "combined": "Combined (ours)", "prompt_guard": "Prompt Guard 2 (Meta)"}
    cols = {"heuristic": AMBER, "judge": BLUE, "combined": TEAL, "prompt_guard": PLUM}
    series = {names[d]: [_pct(bench["by_family"][f][d]) for f in fams] for d in dets}
    series_fp = {names[d]: [_pct(bench["fpr_clean"][d]), _pct(bench["fpr_hard"][d])] for d in dets}
    fig_grouped([f.title() for f in fams] + ["All payloads"], {k: v + [_pct(bench["tpr"][d])] for (k, v), d in zip(series.items(), dets)},
                [cols[d] for d in dets], "fig_injection_bench.png", "Injection detection rate by payload family", "Detected (%)")
    fig_grouped(["Clean corpus docs", "Hard negatives"], series_fp, [cols[d] for d in dets], "fig_injection_fpr.png",
                "False-positive rate on benign text", "Flagged (%)")


def fig_dpia(comp_json):
    risks = comp_json["risks"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, r in enumerate(risks[::-1]):
        a, b = r["inherent"][2], r["residual"][2]
        ax.plot([b, a], [i, i], color=GREY, linewidth=2.5, zorder=1, solid_capstyle="round")
        ax.scatter([a], [i], color=AMBER, s=46, zorder=3, edgecolor="white", linewidth=1.2, label="Inherent" if i == 0 else None)
        ax.scatter([b], [i], color=TEAL, s=46, zorder=3, edgecolor="white", linewidth=1.2, label="Residual" if i == 0 else None)
    ax.set_yticks(range(len(risks))); ax.set_yticklabels([f"{r['id']} {r['threat'][:42]}" for r in risks[::-1]], fontsize=7.5)
    ax.set_xlim(0, 26); ax.set_xlabel("Risk score = likelihood × impact (1–25, expert judgement)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.set_title("DPIA: inherent vs residual risk", loc="left", fontsize=10)
    fig.savefig(OUT / "fig_dpia.png"); plt.close(fig)


def _box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")


def _arrow(ax, x1, y1, x2, y2, label=None, color=MUTED, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, ls=ls))
    if label:
        ax.text((x1 + x2) / 2, max(y1, y2) + 0.42, label, ha="center", fontsize=6.3, color=MUTED,
                bbox=dict(fc="white", ec="none", pad=0.6))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.9); ax.axis("off")
    soft, edge = "#EAF4F0", TEAL
    _box(ax, 0.05, 2.75, 1.25, 0.7, "User prompt\n+ role + purpose", "#FFF6EA", AMBER)
    _box(ax, 1.65, 2.75, 1.5, 0.7, "① PII redaction\n(prompt)", soft, edge, bold=True)
    _box(ax, 3.5, 2.75, 1.55, 0.7, "Redacted index\n(TF-IDF / bge-small)", "#EEF1F8", BLUE)
    _box(ax, 5.4, 2.75, 1.7, 0.7, "② Purpose-based\naccess filter", soft, edge, bold=True)
    _box(ax, 7.45, 2.75, 2.45, 0.7, "③ Injection screening\nheuristic + LLM judge", soft, edge, bold=True)
    _box(ax, 7.45, 1.55, 2.45, 0.7, "① PII redaction\n(surviving chunks)", soft, edge, bold=True)
    _box(ax, 5.4, 1.55, 1.7, 0.7, "LLM\n(Groq / extractive)", "#F3F3F1", MUTED)
    _box(ax, 3.5, 1.55, 1.55, 0.7, "Answer to user", "#FFF6EA", AMBER)
    _box(ax, 0.05, 1.3, 2.6, 1.0, "Enterprise corpus\n(email · HR · tickets · code)\nredacted before indexing", "#EEF1F8", BLUE, fs=7)
    _box(ax, 0.6, 0.12, 8.8, 0.62, "④ Hash-chained audit log — every redaction, drop and judge score (types and counts only, never values)",
         "#F5F0F7", PLUM, bold=True)
    _arrow(ax, 1.3, 3.1, 1.65, 3.1); _arrow(ax, 3.15, 3.1, 3.5, 3.1, "query")
    _arrow(ax, 5.05, 3.1, 5.4, 3.1, "candidates"); _arrow(ax, 7.1, 3.1, 7.45, 3.1, "in-scope")
    _arrow(ax, 8.67, 2.75, 8.67, 2.25); _arrow(ax, 7.45, 1.9, 7.1, 1.9)
    _arrow(ax, 5.4, 1.9, 5.05, 1.9); _arrow(ax, 2.0, 2.3, 3.6, 2.75, color=BLUE, ls="--")
    ax.text(2.45, 2.6, "redacted ingestion", fontsize=6.3, color=BLUE, rotation=14)
    for x, y0 in ((2.4, 2.75), (7.27, 2.75), (8.67, 1.55)):      # audit taps run through gaps, never through boxes
        _arrow(ax, x, y0, x, 0.74, color=PLUM, ls=":")
    fig.savefig(OUT / "fig_architecture.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comp = load("comparison.json")
    fig_architecture()
    fig_comparison(comp, [("context_pii_exposure", "PII sent to LLM provider"), ("oversharing_rate", "Over-sharing"),
                          ("injection_exposure_rate", "Injection exposure"), ("answer_quality_rouge_l", "Answer quality (ROUGE-L)")],
                   ["none", "rbac", "pii_only", "output_filter", "spotlight", "heuristic", "full"],
                   "fig_comparison.png", "Same corpus, queries and LLM — only the defence differs")
    fig_comparison(comp, [("context_pii_exposure", "PII sent to LLM provider"), ("oversharing_rate", "Over-sharing"),
                          ("index_pii_exposure", "PII in vector index"), ("answer_quality_rouge_l", "Answer quality (ROUGE-L)")],
                   ["full", "full_no_judge", "full_no_access", "full_no_pii", "full_no_index", "full_dense"],
                   "fig_ablation.png", "Ablation — remove one control at a time")
    fig_privacy_utility(comp)
    fig_pii(load("pii_detection.json"))
    fig_bench(load("injection_bench.json"))
    fig_dpia(load("compliance.json"))
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
