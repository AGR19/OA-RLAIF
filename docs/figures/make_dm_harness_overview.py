"""The dm_harness overview figure (SVG): the three arms, the calibration of arm C, the three confidence scores, an
illustration of the bin rule, and the Top-k scoring."""
import sys
import textwrap
from html import escape

OUT = sys.argv[1]
W, H = 1400, 1620
INK, MUTED, LINE, BG = "#2C2C2A", "#5F5E5A", "#B4B2A9", "#FFFFFF"
A = ("#185FA5", "#E6F1FB")  # arm A: blue
B = ("#BA7517", "#FAEEDA")  # arm B: amber
C = ("#0F6E56", "#E1F5EE")  # arm C: teal
G = ("#5F5E5A", "#F1EFE8")  # neutral
RED = "#A32D2D"
FONT = "Helvetica, Arial, sans-serif"
svg = []


def text(x, y, s, size=14, weight="normal", color=INK, anchor="start", italic=False):
    style = ' font-style="italic"' if italic else ""
    svg.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}" '
               f'text-anchor="{anchor}"{style}>{escape(s)}</text>')


def lines(x, y, rows, size=13, lead=None, color=INK, anchor="start", weight="normal"):
    lead = lead or size * 1.35
    for i, r in enumerate(rows):
        text(x, y + i * lead, r, size, weight, color, anchor)
    return y + len(rows) * lead


def wrap(s, width, size):
    return textwrap.wrap(s, max(8, int(width / (size * 0.58))))


def box(x, y, w, h, colors, title=None, body=None, size=13, center=False, title_size=14, rx=8):
    stroke, fill = colors
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="1.5"/>')
    cx = x + w / 2 if center else x + 12
    anchor = "middle" if center else "start"
    ty = y + 22
    if title:
        for r in wrap(title, w - 24, title_size):
            text(cx, ty, r, title_size, "bold", stroke, anchor)
            ty += title_size * 1.3
        ty += 2
    if body:
        for para in body if isinstance(body, list) else [body]:
            ty = lines(cx, ty, wrap(para, w - 24, size), size, anchor=anchor)


def arrow(x1, y1, x2, y2, color=MUTED, label=None, lx=None, ly=None, dash=False):
    d = ' stroke-dasharray="5 4"' if dash else ""
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6" '
               f'marker-end="url(#head)"{d}/>')
    if label:
        text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 6, label, 12, "bold",
             color, "middle")


def diamond(cx, cy, w, h, colors, rows):
    stroke, fill = colors
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    svg.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines(cx, cy - (len(rows) - 1) * 8 + 5, rows, 13, 16, anchor="middle", weight="bold", color=stroke)


def section(y, s):
    text(40, y, s, 18, "bold")
    svg.append(f'<line x1="40" y1="{y + 8}" x2="{W - 40}" y2="{y + 8}" stroke="{LINE}" stroke-width="1"/>')


# ---- header -----------------------------------------------------------------------------------------------------------
text(40, 44, "How pplx-decider controls the All-merged agent through JevAgentHarness (branch dm_harness)", 24, "bold")
lines(40, 72, ["Each arm aligns all 299,961 classes: every class with a reference partner plus the complete SNOMED CT → "
               "ICD-11 map. Nothing is held out. The calibration sample of arm C is",
               "also aligned and scored. Score: Top-k (k = 1, 3, 5) on the 270,290 classes that have a reference "
               "partner. Compared with All merged and All merged + agent from oa-rlaif-full-eval-results (read only)."],
      14, color=MUTED)

# ---- 1. the three arms ------------------------------------------------------------------------------------------------
section(134, "1 · The three arms (one class = one episode)")
box(40, 156, 230, 170, G, "One episode", ["The source class (name, synonyms, parents)", "Candidates C1 … Cn from "
                                          "retrieval, + NONE", "Both ontologies' modelling profiles"])
for y in (215, 410, 610):
    arrow(270, 241, 300, y if y != 241 else y, color=LINE)

# arm A
svg.append(f'<rect x="300" y="160" width="1060" height="110" rx="10" fill="none" stroke="{A[0]}" stroke-width="2"/>')
text(316, 184, "Arm A · pplx-decider alone (tag decider)", 15, "bold", A[0])
box(316, 196, 300, 62, A, None, ['One "choice" question: "Which candidate is the same concept as the source?" '
                                 'Options C1 … Cn, NONE'])
arrow(616, 227, 660, 227, A[0])
box(660, 196, 300, 62, A, None, ["Probabilities over the options; confidence = the probability of its pick"])
arrow(960, 227, 1004, 227, A[0])
box(1004, 196, 340, 62, A, None, ["Answer = the most likely option. Top-k ranks 2–5 follow its probabilities."])

# arm B
svg.append(f'<rect x="300" y="286" width="1060" height="272" rx="10" fill="none" stroke="{B[0]}" stroke-width="2"/>')
text(316, 310, "Arm B · JevAgentHarness + pplx-decider + All merged + mini-SWE-agent, fixed τ (tag jah-agent)", 15,
     "bold", B[0])
box(316, 324, 210, 96, B, None, ["mini-SWE-agent runs om profile, om show, om candidates, then asks the model"])
arrow(526, 372, 560, 372, B[0])
box(560, 324, 230, 96, B, None, ['Before each model call the proxy asks pplx-decider: "Is the next step routine '
                                 'or hard?"'])
arrow(790, 372, 822, 372, B[0])
diamond(902, 372, 160, 92, B, ["p(hard)", "≥ τ_r = 0.75 ?"])
arrow(982, 372, 1030, 344, B[0], "yes", 1000, 346)
arrow(982, 372, 1030, 400, B[0], "no", 1000, 408)
box(1030, 320, 150, 48, B, None, ["Strong tier:", "thinking ON"], 12)
box(1030, 376, 150, 48, B, None, ["Fast tier:", "thinking OFF"], 12)
lines(1192, 336, ["Same All-merged", "weights on vLLM;", "the tier sets", "enable_thinking.", "A decision error",
                  "→ strong tier."], 12, 16, color=MUTED)
# the guard
text(316, 450, "The NONE / invalid guard (the paper's GUARD mode, in dm_harness/harness.py)", 13, "bold", B[0])
box(316, 460, 190, 86, B, None, ["om submit: a match ends the class. NONE or invalid → the guard."], 12)
arrow(506, 503, 540, 503, B[0])
box(540, 460, 230, 86, B, None, ["pplx-decider answers the candidate question: p(best) = probability of the most "
                                 "likely candidate"], 12)
arrow(770, 503, 802, 503, B[0])
diamond(870, 503, 136, 80, B, ["p(best)", "≥ τ_g = 0.5 ?"])
arrow(938, 503, 980, 481, B[0], "yes", 958, 478)
arrow(938, 503, 980, 527, B[0], "no", 958, 538)
box(980, 458, 370, 46, B, None, ["Redo on the strong tier with a hint:", '"candidate Cx may be the target"'], 12)
box(980, 506, 370, 46, B, None, ["NONE stays. Invalid: redo without the hint;", "still invalid → pplx-decider's pick"],
    12)

# arm C
svg.append(f'<rect x="300" y="574" width="1060" height="124" rx="10" fill="none" stroke="{C[0]}" stroke-width="2"/>')
text(316, 598, "Arm C · the same stack with thresholds fitted on a sample, plus the cascade (tag jah-fit-agent)", 15,
     "bold", C[0])
diamond(420, 650, 200, 70, C, ["arm-A confidence", "≥ τ_c ?"])
arrow(520, 650, 566, 628, C[0], "yes", 540, 626)
arrow(520, 650, 566, 670, C[0], "no", 540, 684)
box(566, 610, 780, 36, C, None, ["CASCADE: pplx-decider's answer is final. Gemma is not asked."], 12)
box(566, 652, 780, 36, C, None, ["Else: arm B's flow with the FITTED τ_r and τ_g (section 2); no τ fitted → that "
                                 "decision is never made."], 12)

# ---- 2. calibration ----------------------------------------------------------------------------------------------------
section(740, "2 · How arm C's thresholds are fitted, then frozen (Zhu et al. 2026, §2.1 and App. A)")
steps = [
    ("1. Sample", ["3,000 scored classes (DM_CAL_N), seed 13, stratified by task, at least 20 per task"]),
    ("2. Shadow runs", ["The decision model answers, never acts:", "• agent, every step thinking ON, p(hard) logged",
                        "• agent, every step thinking OFF", "• each NONE / invalid redone with the hint",
                        "• pplx-decider's own answer"]),
    ("3. Right or wrong", ["A run is right when its first answer is a reference partner of the class"]),
    ("4. Equal-mass bins", ["Sort each decision point's runs by its score; bins of ≥ 20 (at most 10); mean gain per "
                            "bin (section 3)"]),
    ("5. τ* and freeze", ["τ* = the lowest score from which EVERY higher bin has gain ≥ 0. No such bin → never act. "
                          "Saved to calibration/thresholds.json"]),
]
x = 40
for i, (t, body) in enumerate(steps):
    box(x, 760, 236, 196, C, t, body, 12)
    if i < len(steps) - 1:
        arrow(x + 236, 858, x + 260, 858, C[0])
    x += 260
arrow(1158, 956, 1158, 976, C[0])
box(40, 976, 1320, 34, C, None, ["Arm C then aligns all 299,961 classes with the frozen τ (the sample included). "
                                 "Table 11 also scores it without the sample."], 13)

# ---- 3. the confidence scores -----------------------------------------------------------------------------------------
section(1050, "3 · The three confidence scores and when they act")
cards = [
    ("Routing · TRIGGER", "p(hard)", "pplx-decider's probability that the next agent step is hard",
     "Strong tier (thinking ON) when p(hard) ≥ τ_r", "B: 0.75, fixed  ·  C: fitted",
     "gain = right(ON) − right(OFF) − ΔC/V"),
    ("Guard · GUARD", "p(best)", "probability of the most likely candidate (NONE excluded), asked after NONE / invalid",
     "Redo on the strong tier with the hint when p(best) ≥ τ_g", "B: 0.5, fixed  ·  C: fitted",
     "gain = right(redo) − right(kept) − ΔC/V"),
    ("Cascade · CASCADE (Eq. 3)", "confidence", "probability of pplx-decider's own pick in arm A",
     "pplx-decider answers alone when confidence ≥ τ_c", "B: not used (Table 8: what-if)  ·  C: fitted",
     "advantage = right(decider) − right(agent) + ΔC/V"),
]
x = 40
for title, score, meaning, fires, taus, gain in cards:
    svg.append(f'<rect x="{x}" y="1070" width="430" height="190" rx="8" fill="{G[1]}" stroke="{G[0]}" '
               f'stroke-width="1.2"/>')
    text(x + 14, 1094, title, 15, "bold")
    text(x + 14, 1122, "Score: " + score, 14, "bold", B[0])
    y = lines(x + 14, 1142, wrap(meaning, 400, 13), 13, color=MUTED)
    y = lines(x + 14, y + 6, ["Acts: " + fires], 13)
    y = lines(x + 14, y + 2, [taus], 13, weight="bold", color=C[0])
    lines(x + 14, y + 2, [gain], 13, color=INK)
    x += 445
lines(40, 1284, ["ΔC = the GPU cost of one agent step (the machine's price per hour ÷ 3,600 ÷ steps per second, measured "
                 "in arm B). V = $1 per wrong class. right(·) = 1 if the first answer is a reference partner, else 0."],
      13, color=MUTED)

# ---- 4. the bin rule (illustration) -----------------------------------------------------------------------------------
section(1326, "4 · Reading τ* off the bins (illustration, not measured data)")
gains = [-0.20, -0.12, -0.05, 0.03, -0.02, 0.04, 0.10, 0.18]
froms = [0.05, 0.18, 0.31, 0.44, 0.55, 0.66, 0.77, 0.88]
ox, oy, bw, scale = 90, 1470, 64, 260
svg.append(f'<line x1="{ox - 10}" y1="{oy}" x2="{ox + 8 * (bw + 10) + 10}" y2="{oy}" stroke="{MUTED}" stroke-width="1"/>')
tau_i = 5
for i, (g, f) in enumerate(zip(gains, froms)):
    x = ox + i * (bw + 10)
    h = abs(g) * scale
    color = C[0] if i >= tau_i else (RED if g < 0 else LINE)
    svg.append(f'<rect x="{x}" y="{oy - h if g > 0 else oy}" width="{bw}" height="{h}" fill="{color}" opacity="0.85"/>')
    text(x + bw / 2, oy + (h + 16 if g < 0 else 16), f"{g:+.2f}", 11, color=MUTED, anchor="middle")
    text(x + bw / 2, 1590, f"from {f:.2f}", 11, color=MUTED, anchor="middle")
tx = ox + tau_i * (bw + 10) - 5
svg.append(f'<line x1="{tx}" y1="1360" x2="{tx}" y2="1574" stroke="{C[0]}" stroke-width="2" stroke-dasharray="6 4"/>')
text(tx + 6, 1372, "τ* = 0.66", 13, "bold", C[0])
text(ox - 10, 1360, "mean gain", 12, color=MUTED)
lines(720, 1366, ["Bins sorted by score (low → high). Scan from the top bin down:",
                  "• bins 8, 7, 6 have gain ≥ 0 → keep going;",
                  "• bin 5 is negative → stop. τ* = where bin 6 starts.",
                  "Bin 4 is positive, but it is below a losing bin, so it does not act.",
                  "If the top bin were negative: no τ, the decision is never made.",
                  "", "Scoring (DRAL-OA clf.py), per scored class c, F_c = top k answers, R_c = partners:",
                  "P@k = |F_c ∩ R_c| / min(k, |R_c|)", "R@k = |F_c ∩ R_c| / |R_c|",
                  "Averaged over classes; F@k = 2·P·R / (P + R); k = 1, 3, 5."], 13)

head = (f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>')
size = "" if len(sys.argv) > 2 else f' width="{W}" height="{H}"'
out = (f'<svg xmlns="http://www.w3.org/2000/svg"{size} viewBox="0 0 {W} {H}">{head}'
       f'<rect width="{W}" height="{H}" fill="{BG}"/>' + "\n".join(svg) + "</svg>\n")
open(OUT, "w").write(out)
print(OUT)
