#!/usr/bin/env python3
"""
One place that decides how big things are in every figure we publish.

THE CANVAS IS THE PAGE. A journal reproduces a figure at a fixed column width,
so the only size that matters is the size ON THE PAGE — and drawing at some
comfortable screen size and letting the publisher scale it down hides that
completely. The sampling-design figure was 20 inches wide with 15 pt labels;
reproduced in a 7.2 inch double column that is 5.4 pt, under every journal's
6-8 pt floor, and its 0.8 pt axis spines land at 0.29 pt. Nothing about the
figure looked wrong until it was printed.

So figures are drawn at the width they will be printed at, in real points. What
you see is what prints. Making the text bigger means a bigger FONT, never a
bigger canvas.

    from figstyle import manuscript, WIDTH
    k = manuscript(WIDTH["double"], base_pt=8)
    fig = plt.figure(figsize=(WIDTH["double"], WIDTH["double"] * 0.65))

`k` is the linear scale from the 20-inch canvas the existing layouts were tuned
on, for the handful of sizes matplotlib states in points but which should shrink
with the page — marker areas go as k**2, line widths as k.
"""

# Standard reproduction widths, inches. Most journals sit within a few mm of
# these; AGU, Copernicus and Elsevier all take 190 mm as the full width.
WIDTH = {"single": 3.5, "double": 7.2, "full": 7.5}

_LAID_OUT_AT = 20.0        # inches; what the panel arrangements were tuned on


def manuscript(width_in=WIDTH["double"], base_pt=8.0):
    """Set rcParams for a figure printed at `width_in`. Returns the size scale.

    base_pt is the body text size on the page. 8 pt suits a double column; drop
    to 7 for a single column, raise to 10-11 for a talk. Ticks go one point
    smaller and titles one larger, which is the usual typographic step.
    """
    import matplotlib.pyplot as plt

    k = width_in / _LAID_OUT_AT
    plt.rcParams.update({
        "font.size": base_pt,
        "axes.titlesize": base_pt + 1,
        "axes.labelsize": base_pt,
        "xtick.labelsize": base_pt - 1,
        "ytick.labelsize": base_pt - 1,
        "legend.fontsize": base_pt - 1,
        # Line art thinner than ~0.5 pt drops out of print and out of a PDF
        # viewer at 100%, so the floor is enforced rather than scaled past.
        "axes.linewidth": max(0.5, 1.8 * k),
        "xtick.major.width": max(0.5, 1.8 * k),
        "ytick.major.width": max(0.5, 1.8 * k),
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": max(0.5, 2.0 * k),
        "patch.linewidth": max(0.4, 2.0 * k),
        "legend.frameon": False,
        # 600 dpi: journals ask 300 for photographs and 600 for anything with
        # text or lines in it, which is everything here.
        "savefig.dpi": 600,
        "figure.dpi": 600,
        "savefig.bbox": None,        # constrained_layout already did the work
    })
    return k
