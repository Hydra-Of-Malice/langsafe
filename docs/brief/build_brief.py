"""Build the one-page mentor brief: python docs/brief/build_brief.py docs/brief/mentor_brief.pdf"""
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

F = "C:/Windows/Fonts/"
for n, f in (("UI", "segoeui"), ("UI-B", "segoeuib"), ("UI-I", "segoeuii"), ("UI-BI", "segoeuiz"), ("UI-L", "segoeuil")):
    pdfmetrics.registerFont(TTFont(n, F + f + ".ttf"))
registerFontFamily("UI", normal="UI", bold="UI-B", italic="UI-I", boldItalic="UI-BI")

INK, MUTED, ACCENT = colors.HexColor("#1f2933"), colors.HexColor("#5b6670"), colors.HexColor("#0f5c6e")
LIGHT, RULE = colors.HexColor("#e6f1f3"), colors.HexColor("#c9d3d8")

body = ParagraphStyle("b", fontName="UI", fontSize=8.6, leading=11.6, textColor=INK, spaceAfter=3)
cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10.4, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="UI-B")
h = ParagraphStyle("h", fontName="UI-B", fontSize=10, leading=13, textColor=ACCENT, spaceBefore=5, spaceAfter=2)
bul = ParagraphStyle("bl", parent=body, leftIndent=10, bulletIndent=1, spaceAfter=1.5)

story = []
P = lambda t, s=body: story.append(Paragraph(t, s))
H = lambda t: story.append(Paragraph(t, h))


def B(items):
    for t in items:
        story.append(Paragraph(t, bul, bulletText="•"))


def T(rows, widths, header=True):
    data = [[Paragraph(c, cellb if header and i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 3))


P("LEXSI LABS AI SAFETY &amp; ALIGNMENT HACKATHON · TRACK 2b: REPAIRING SAFETY DRIFT WITHOUT RETRAINING",
  ParagraphStyle("k", parent=body, fontName="UI-B", fontSize=7.4, textColor=ACCENT, spaceAfter=1))
P("Same model, different language, different safety",
  ParagraphStyle("t", fontName="UI-B", fontSize=16, leading=20, textColor=INK, spaceAfter=1))
P("Team: Aditya Arnav, Pallab Das · Mentor brief, 7 November 2026 · Core model: Tiny Aya Earth (3.35B), "
  "reference Tiny Aya Global",
  ParagraphStyle("m", parent=body, fontName="UI-L", fontSize=8.2, textColor=MUTED, spaceAfter=4))

H("The problem")
P("Tiny Aya's regional models average almost the same safety as Global, so a check on averages finds nothing. The tech "
  "report gives one clue: each model's <b>worst</b> language on MultiJail is 87.0 for Global, 78.1 for Fire and 77.5 for "
  "Earth (Table 7), and it doesn't name that language. We set out to find it. Our first pilot looked in the models' own "
  "regions and found no gap, because we were scoring refusals. Scored the report's way, by answer content, the gap appears "
  "in a language neither model was tuned for.")

H("What the pilot found (planning signal, not a result)")
T([["Javanese, safe answers", "Global", "Fire", "Earth", "Water", "10-language mean (G / F / E / W)"],
   ["With Tiny Aya's preamble", "87.0%", "80.9%", "77.1%", "92.4%", "94.9 / 94.1 / 91.6 / 94.9"],
   ["Without the preamble", "91.6%", "80.9% (p=0.003)", "<b>74.8%</b> (p&lt;0.001)", "89.3%", "95.0 / 94.6 / 93.5 / 94.2"]],
  [37 * mm, 14 * mm, 27 * mm, 28 * mm, 14 * mm, 50 * mm])
P("131 prompts per language, repair split only, 4-bit weights, two local 8B judges standing in for Command A. Without the "
  "preamble Earth's gap survives correction for all 60 comparisons, and Fire's only for 10 languages. Water shows no Javanese "
  "gap, which fits a gap that follows a model's region, though placing Javanese in Water's region is our reading of its card. "
  "Water dips in Bengali instead (85.5% and 84.7%, not significant). <b>Caution:</b> both judges "
  "call only 3 or 4 of Earth's 131 Javanese answers unsafe. The rest is judge disputes and breakdown: a blind read of 153 "
  "answers found many are repetition loops (19.8% of Earth's, 7.6% of Global's). The true gap is unknown until held-out data.")

H("What we are doing")
T([["Step", "What happens", "Lexsi kit"],
   ["1. Data", "Harmful prompts (Aya red-teaming, MultiJail) and harmless look-alikes (XSTest) in English, Arabic, Swahili, "
               "Javanese, Bengali, Thai and Korean. Splits are by item, so a translation of a test prompt can never be used "
               "for the repair.", "CuratorKIT"],
   ["2. Confirm", "At full precision on held-out prompts, per language: content safety, wrongful refusal of harmless "
                  "prompts, and general ability. No repair claim unless the gap passes a corrected test.", "AuditKIT"],
   ["3. Locate", "EAP-IG finds the safety circuit per language and per model. Where did regional tuning change it?",
    "CircuitKIT"],
   ["4. Repair", "Five edits compared head to head, with settings tuned on the repair split only.", "SafeTune"],
   ["5. Verify", "Re-test all three axes, then again after 4-bit compression (NF4 and GGUF Q4_K_M).", "AuditKIT"]],
  [17 * mm, 133 * mm, 20 * mm])

H("The repair comparison")
T([["Arm", "Edit", "Why it is there"],
   ["ReSta / Merge-back", "Earth + α·(Global − Base), or a blend toward Global, whole model", "Weight-space baselines"],
   ["<b>Circuit arms</b>", "<b>The same two vectors, only in the top 20% of layers by CircuitKIT score</b>",
    "<b>Our main bet. Paired with its whole-model twin, so the difference isolates what the map adds</b>"],
   ["Random layers", "The best circuit edit on random layers outside the circuit, 3 seeds",
    "The control: does the map beat editing fewer layers?"],
   ["Steering", "Refusal direction added at run time", "Known to cause over-refusal; we check on Tiny Aya"]],
  [30 * mm, 70 * mm, 70 * mm])

H("How we will judge success (fixed before any repair)")
B(["The gap must first be confirmed: full precision, held-out prompts, each preamble condition, paired CI above zero, "
   "corrected p below 0.05.",
   "Then close at least <b>half</b> of it, with wrongful refusal up by <b>at most 2 points</b> in any language and ability "
   "down by <b>at most 1 point</b>, all still holding at 4-bit.",
   "Every arm is reported, including losing ones and negative results."])

H("Where we are today")
P("The pipeline has run end to end on real Earth and Fire models at a small sample, and those runs fixed several crashes and "
  "silent errors. Next: confirm the gap on held-out data, with Fire and Water alongside as context, then run the repairs. We "
  "have also prepared six fixes for the kits (none submitted yet).")

H("What we would value your view on")
B(["Is content safety, as in the report, the right primary measure? And should a model that breaks down into repetition "
   "count as a safety failure, as the report's rubric implies, or be reported only as a separate failure?",
   "Do you know which language is behind the report's 78.1, and are our 8B judges a reasonable stand-in for Command A?",
   "Is a public multilingual eval set (prompts and labels only) acceptable under Cohere's acceptable-use policy?"])

doc = SimpleDocTemplate(sys.argv[1], pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=13 * mm,
                        bottomMargin=11 * mm, title="Same model, different language, different safety",
                        author="Aditya Arnav, Pallab Das", subject="Mentor brief, Lexsi hackathon")
doc.build(story)
print("ok", sys.argv[1])
