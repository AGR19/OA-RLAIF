"""Charts of Top-k (k = 1, 3, 5) scores: Precision vs. Top-k, Recall vs. Top-k and F-measure vs. Top-k, one line per
model, the layout of the Advanced-Cybersecurity-OA figures. Plain SVG (no plotting package needed).

    write_topk_chart(path, {"GRPO": topk_result, "SFT": topk_result}, "Anatomy (mouse -> human)")
where topk_result is AlignmentEvaluator.topk(...)["k"] (or the whole dict).
"""

from __future__ import annotations

from html import escape
from pathlib import Path

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#bcbd22"]
MARKERS = ["circle", "square", "triangle", "diamond", "circle", "square", "triangle", "diamond", "circle", "square"]
PANELS = [("Precision", "Precision vs. Top-k"), ("Recall", "Recall vs. Top-k"), ("F-measure", "F-measure vs. Top-k")]


def _marker(kind: str, x: float, y: float, color: str) -> str:
    if kind == "square":
        return f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{color}"/>'
    if kind == "triangle":
        return f'<polygon points="{x:.1f},{y - 5:.1f} {x - 5:.1f},{y + 4:.1f} {x + 5:.1f},{y + 4:.1f}" fill="{color}"/>'
    if kind == "diamond":
        return f'<polygon points="{x:.1f},{y - 5:.1f} {x + 5:.1f},{y:.1f} {x:.1f},{y + 5:.1f} {x - 5:.1f},{y:.1f}" fill="{color}"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'


def write_topk_chart(path, series: dict[str, dict], title: str) -> Path:
    """series: label -> {k: {"Precision", "Recall", "F-measure"}} (keys may be ints or strings)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {label: {int(k): v for k, v in (s.get("k", s)).items()} for label, s in series.items() if s}
    ks = sorted({k for r in rows.values() for k in r}) or [1, 3, 5]
    values = [r[k][m] for r in rows.values() for k in r for m, _ in PANELS]
    low = min([0.8] + values)
    low = max(0.0, (int(low * 20) / 20) - 0.05)  # round down to 0.05, a little margin
    width, panel_h, top, left, right, gap = 760, 230, 50, 70, 230, 40
    height = top + len(PANELS) * (panel_h + gap) + 10
    plot_w = width - left - right
    xs = {k: left + plot_w * (i + 0.5) / len(ks) for i, k in enumerate(ks)}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           'font-family="Helvetica, Arial, sans-serif">', f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-size="16" font-weight="bold">{escape(title)}</text>']
    for p, (metric, panel_title) in enumerate(PANELS):
        y0 = top + p * (panel_h + gap)
        y = lambda v: y0 + 20 + (panel_h - 40) * (1 - (v - low) / (1 - low or 1))  # noqa: E731
        out.append(f'<text x="{left + plot_w / 2:.0f}" y="{y0 + 12}" text-anchor="middle" font-size="13">{panel_title}</text>')
        out.append(f'<rect x="{left}" y="{y0 + 20}" width="{plot_w}" height="{panel_h - 40}" fill="none" stroke="#999"/>')
        steps = 5
        for i in range(steps + 1):
            v = low + (1 - low) * i / steps
            out.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#eee"/>')
            out.append(f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="10">{v:.2f}</text>')
        for k in ks:
            out.append(f'<text x="{xs[k]:.1f}" y="{y0 + panel_h - 4}" text-anchor="middle" font-size="11">Top-{k}</text>')
        out.append(f'<text x="18" y="{y0 + panel_h / 2:.0f}" font-size="11" transform="rotate(-90 18 {y0 + panel_h / 2:.0f})" '
                   f'text-anchor="middle">{metric}</text>')
        for i, (label, r) in enumerate(rows.items()):
            color, marker = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
            points = [(xs[k], y(r[k][metric])) for k in ks if k in r]
            out.append('<polyline fill="none" stroke-width="2" stroke="' + color + '" points="'
                       + " ".join(f"{a:.1f},{b:.1f}" for a, b in points) + '"/>')
            out += [_marker(marker, a, b, color) for a, b in points]
    for i, label in enumerate(rows):  # legend
        ly = top + 30 + i * 20
        color, marker = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
        out.append(f'<line x1="{width - right + 20}" x2="{width - right + 44}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        out.append(_marker(marker, width - right + 32, ly, color))
        out.append(f'<text x="{width - right + 52}" y="{ly + 4}" font-size="11">{escape(label)}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")
    return path
