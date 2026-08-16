#!/usr/bin/env python3
"""analysis.json → a slide deck
src/agents/analysis/step4_slides.py

    in   the report dict step4_report.build() assembled
    out  04_analysis/analysis.pptx

A SECOND RENDERING, NOT A SECOND REPORT. analysis.json stays the Analyzer's
one authoritative output; this reads it and lays it out. Nothing here computes,
rounds, re-words or omits a number — a deck that disagreed with the file it came
from would be the worst artifact in the run, because it is the one that gets
presented while the file is the one that gets checked.

WHAT THAT RULES OUT, concretely: no claim appears without the finding id it
rests on; the struck claims travel, with the reason; the caveats are quoted
rather than summarised; and a run whose status is not `supported` says so on
the title slide rather than opening with an answer it has not earned.

python-pptx IS OPTIONAL. It is present in the ideas env, and a run without it
must still finish — the deck is a convenience, and losing it is not losing the
analysis. `build()` returns None and says why.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

FILENAME = "analysis.pptx"

# 16:9 in EMU, the unit python-pptx measures in. 914400 EMU = 1 inch.
_EMU_IN = 914400
SLIDE_W = int(13.333 * _EMU_IN)
SLIDE_H = int(7.5 * _EMU_IN)

# Status → what the title slide says about itself. A deck that opens with an
# answer for a run that failed is the failure this whole week has been about.
_STATUS_LINE = {
    "supported":           ("", 0x1F, 0x6F, 0x5C),
    "no_supported_claims": ("No claim survived the audit — the reasons are at "
                            "the end.", 0x8A, 0x64, 0x10),
    "insufficient_evidence": ("The evidence was judged insufficient. This is a "
                              "result, not a malfunction.", 0x8A, 0x64, 0x10),
    "analysis_failed":     ("THE ANALYSIS DID NOT COMPLETE — the steps that "
                            "failed are named below.", 0xA6, 0x34, 0x2B),
}


def _text(slide, left, top, width, height, size, text,
          bold=False, rgb=(0x14, 0x1D, 0x1C), wrap=True):
    from pptx.util import Emu, Pt
    from pptx.dml.color import RGBColor
    box = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(width), Emu(height))
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.text = str(text)
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = RGBColor(*rgb)
    return box


def _bullets(slide, left, top, width, height, items, size=14):
    """One paragraph per item; (text, indent) pairs keep sub-lines attached."""
    from pptx.util import Emu, Pt
    from pptx.dml.color import RGBColor
    box = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(width), Emu(height))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        text, indent = item if isinstance(item, tuple) else (item, 0)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = str(text)
        p.level = min(indent, 4)
        for r in p.runs:
            r.font.size = Pt(size - 2 * indent)
            r.font.color.rgb = RGBColor(0x45, 0x54, 0x57) if indent \
                else RGBColor(0x14, 0x1D, 0x1C)
    return box


def _picture(slide, path, left, top, max_w, max_h):
    """Fit an image inside a box without distorting it. Missing file → None."""
    from pptx.util import Emu
    try:
        from PIL import Image
        with Image.open(path) as im:
            iw, ih = im.size
    except Exception:                                           # noqa: BLE001
        return None
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    try:
        return slide.shapes.add_picture(
            str(path), Emu(left + (max_w - w) // 2),
            Emu(top + (max_h - h) // 2), Emu(w), Emu(h))
    except Exception:                                           # noqa: BLE001
        return None


def build(report: Dict[str, Any], out_dir) -> Optional[str]:
    """Write the deck. Returns its path, or None with a printed reason."""
    try:
        from pptx import Presentation
    except ImportError:
        print("   ⏭  no slide deck — python-pptx is not installed "
              "(pip install python-pptx). analysis.json is unaffected.")
        return None

    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    blank = prs.slide_layouts[6]
    M = int(0.6 * _EMU_IN)                 # margin
    W = SLIDE_W - 2 * M

    status = report.get("status") or "unknown"
    prov = report.get("provenance") or {}
    claims = report.get("claims") or []
    withheld = report.get("withheld") or []
    caveats = report.get("caveats") or []

    # ── 1 · the question and what kind of answer this is ────────────────
    s = prs.slides.add_slide(blank)
    _text(s, M, int(0.8 * _EMU_IN), W, int(0.4 * _EMU_IN), 13,
          "IDEAS · ELM ensemble", rgb=(0x7A, 0x89, 0x8C))
    _text(s, M, int(1.3 * _EMU_IN), W, int(2.4 * _EMU_IN), 26,
          str(report.get("question") or "(no question recorded)"), bold=True)
    note, *rgb = _STATUS_LINE.get(status, ("", 0x45, 0x54, 0x57))
    _text(s, M, int(4.2 * _EMU_IN), W, int(0.5 * _EMU_IN), 15,
          status.replace("_", " "), bold=True, rgb=tuple(rgb))
    if note:
        _text(s, M, int(4.8 * _EMU_IN), W, int(0.9 * _EMU_IN), 13, note,
              rgb=tuple(rgb))
    failed = prov.get("failed_steps")
    if failed:
        _text(s, M, int(5.5 * _EMU_IN), W, int(0.5 * _EMU_IN), 12,
              "steps that did not complete: " + ", ".join(failed),
              rgb=(0xA6, 0x34, 0x2B))
    cost = (report.get("cost") or {})
    comp = (cost.get("compute") or {})
    llm = (cost.get("llm") or {})
    an = llm.get("analyzer") or {}
    spend = ("model spend not measured" if llm.get("measured") is False
             else f"{an.get('calls', 0)} model call(s)")
    _text(s, M, int(6.3 * _EMU_IN), W, int(0.5 * _EMU_IN), 11,
          f"{comp.get('columns_succeeded')}/{comp.get('columns_total')} columns · "
          f"{len(prov.get('rounds') or [])} round(s) · {spend} · "
          f"{report.get('generated_at')}", rgb=(0x7A, 0x89, 0x8C))

    # ── 2 · the answer, verbatim ────────────────────────────────────────
    answer = report.get("answer")
    if answer:
        s = prs.slides.add_slide(blank)
        _text(s, M, M, W, int(0.5 * _EMU_IN), 13, "The answer",
              bold=True, rgb=(0x0E, 0x6E, 0x7A))
        # VERBATIM. Step 3 wrote this and step 4 copied it; a deck that
        # tightened the wording would be a third author nobody audited.
        _text(s, M, int(1.4 * _EMU_IN), W, int(4.5 * _EMU_IN), 18, str(answer))

    # ── 3..n · one slide per claim, with the figure it rests on ─────────
    figs = report.get("figures") or {}
    known = dict(figs.get("comparison") or {})
    known.update(figs.get("investigation") or {})
    for i, c in enumerate(claims, 1):
        s = prs.slides.add_slide(blank)
        _text(s, M, int(0.45 * _EMU_IN), W, int(0.35 * _EMU_IN), 11,
              f"claim {i} of {len(claims)}", rgb=(0x7A, 0x89, 0x8C))
        _text(s, M, int(0.85 * _EMU_IN), W, int(1.1 * _EMU_IN), 17,
              str(c.get("claim") or ""), bold=True)

        pic = c.get("figure") or known.get(c.get("finding_id"))
        if pic and Path(str(pic)).is_file():
            _picture(s, pic, M, int(2.1 * _EMU_IN),
                     int(W * 0.62), int(4.2 * _EMU_IN))

        # EVERY CLAIM CARRIES ITS PROVENANCE. The id is what the audit checked
        # it against, and a slide without it is a sentence with no evidence —
        # which is exactly what the audit exists to remove.
        side_l = M + int(W * 0.65)
        side_w = W - int(W * 0.65)
        lines: List[Any] = [f"finding · {c.get('finding_id')}"]
        if c.get("n") is not None:
            lines.append(f"n = {c['n']}")
        vals = c.get("values") or []
        if vals:
            lines.append("values as measured:")
            lines += [(str(v), 1) for v in vals[:8]]
        if c.get("variables"):
            lines.append("variables: " + ", ".join(str(v) for v in
                                                   c["variables"][:8]))
        for cid in (c.get("caveats") or []):
            lines.append(f"⚠ {cid}")
        _bullets(s, side_l, int(2.1 * _EMU_IN), side_w, int(4.2 * _EMU_IN),
                 lines, size=12)

    # ── caveats ─────────────────────────────────────────────────────────
    blocking = [c for c in caveats if c.get("severity") == "blocking"]
    if blocking:
        s = prs.slides.add_slide(blank)
        _text(s, M, M, W, int(0.5 * _EMU_IN), 13,
              f"What these results may not be used for "
              f"({len(blocking)} blocking of {len(caveats)})",
              bold=True, rgb=(0xA6, 0x34, 0x2B))
        items: List[Any] = []
        for c in blocking[:6]:
            items.append(f"{c.get('id')} — applies to {c.get('applies_to')}")
            items.append((str(c.get("statement"))[:260], 1))
        _bullets(s, M, int(1.4 * _EMU_IN), W, int(5.2 * _EMU_IN), items, 13)

    # ── what was rejected ───────────────────────────────────────────────
    if withheld:
        s = prs.slides.add_slide(blank)
        _text(s, M, M, W, int(0.5 * _EMU_IN), 13,
              f"Claims the audit removed ({len(withheld)})",
              bold=True, rgb=(0x8A, 0x64, 0x10))
        _text(s, M, int(1.0 * _EMU_IN), W, int(0.4 * _EMU_IN), 11,
              "Kept here because a claim that was made and rejected is "
              "evidence about the run.", rgb=(0x7A, 0x89, 0x8C))
        items = []
        for c in withheld[:6]:
            items.append(str(c.get("claim"))[:180])
            items.append((f"[{c.get('struck_by') or 'audit'}] "
                          f"{str(c.get('struck_because'))[:220]}", 1))
        _bullets(s, M, int(1.6 * _EMU_IN), W, int(5.0 * _EMU_IN), items, 12)

    out = Path(out_dir) / FILENAME
    try:
        prs.save(str(out))
    except Exception as e:                                      # noqa: BLE001
        print(f"   ⚠️  slide deck failed ({e}) — analysis.json is unaffected")
        return None
    return str(out)
