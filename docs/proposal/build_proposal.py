"""Build the Candidate A mentor proposal PDF."""
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                Table, TableStyle, PageBreak, KeepTogether, Flowable)

OUT = sys.argv[1]

F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("UI", F + "segoeui.ttf"))
pdfmetrics.registerFont(TTFont("UI-B", F + "segoeuib.ttf"))
pdfmetrics.registerFont(TTFont("UI-I", F + "segoeuii.ttf"))
pdfmetrics.registerFont(TTFont("UI-BI", F + "segoeuiz.ttf"))
pdfmetrics.registerFont(TTFont("UI-L", F + "segoeuil.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("UI", normal="UI", bold="UI-B", italic="UI-I", boldItalic="UI-BI")

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#5b6670")
ACCENT = colors.HexColor("#0f5c6e")
ACCENT_LIGHT = colors.HexColor("#e6f1f3")
RULE = colors.HexColor("#c9d3d8")
WARN_BG = colors.HexColor("#fdf6e7")

body = ParagraphStyle("body", fontName="UI", fontSize=10.2, leading=15, textColor=INK, spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.8, leading=12.2, spaceAfter=3)
cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=12.5, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="UI-B")
h1 = ParagraphStyle("h1", fontName="UI-B", fontSize=15, leading=20, textColor=ACCENT, spaceBefore=12, spaceAfter=6, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="UI-B", fontSize=11.5, leading=16, textColor=INK, spaceBefore=8, spaceAfter=4, keepWithNext=1)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=3)
title = ParagraphStyle("title", fontName="UI-B", fontSize=22, leading=28, textColor=INK, spaceAfter=6)
subtitle = ParagraphStyle("subtitle", fontName="UI-L", fontSize=13, leading=18, textColor=MUTED, spaceAfter=14)
quote = ParagraphStyle("quote", parent=body, fontName="UI-I", fontSize=11, leading=16.5, textColor=INK,
                       leftIndent=10, rightIndent=10)
ref = ParagraphStyle("ref", parent=body, fontSize=8.8, leading=12, leftIndent=18, firstLineIndent=-18, spaceAfter=4)

story = []
P = lambda t, s=body: story.append(Paragraph(t, s))
H1 = lambda t: story.append(Paragraph(t, h1))
H2 = lambda t: story.append(Paragraph(t, h2))
def B(items, style=bullet):
    for t in items:
        story.append(Paragraph(t, style, bulletText="•"))
def SP(h=6):
    story.append(Spacer(1, h))


def table(rows, widths, header=True, zebra=True, bold_first_col=False):
    data = []
    for i, r in enumerate(rows):
        row = []
        for j, c in enumerate(r):
            st = cellb if (header and i == 0) or (bold_first_col and j == 0) else cell
            row.append(Paragraph(c, st) if isinstance(c, str) else c)
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), ACCENT_LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 0.9, ACCENT)]
    if zebra:
        for i in range(1 if header else 0, len(data)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f9fa")))
    t.setStyle(TableStyle(style))
    story.append(t)
    SP(8)


def callout(paragraphs, bg=ACCENT_LIGHT, border=ACCENT):
    inner = [Paragraph(p, body) if isinstance(p, str) else p for p in paragraphs]
    t = Table([[inner]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    SP(10)


class Pipeline(Flowable):
    """Six-step pipeline drawn as boxes in two rows with arrows."""
    steps = [
        ("1  CuratorKIT", "Build the multilingual\ntest and repair sets"),
        ("2  AuditKIT", "Measure the drift,\nper language"),
        ("3  CircuitKIT", "Find where refusal\nlives in the model"),
        ("4  SafeTune", "Repair it, four\ncompeting ways"),
        ("5  AuditKIT", "Re-test, then re-test\nagain at 4-bit"),
        ("6  Hugging Face", "Publish everything,\nincluding failures"),
    ]

    def __init__(self, width):
        super().__init__()
        self.width, self.height = width, 62 * mm

    def draw(self):
        c = self.canv
        bw, bh, gap = 50 * mm, 22 * mm, (self.width - 3 * 50 * mm) / 2
        rows_y = [self.height - bh - 4 * mm, 6 * mm]
        pos = []
        for i in range(6):
            r = i // 3
            k = i % 3 if r == 0 else 2 - (i % 3)  # snake layout
            pos.append((k * (bw + gap), rows_y[r]))
        for i, (x, y) in enumerate(pos):
            c.setFillColor(ACCENT_LIGHT if i not in (2, 3) else colors.HexColor("#d3e8ec"))
            c.setStrokeColor(ACCENT)
            c.setLineWidth(0.9)
            c.roundRect(x, y, bw, bh, 3 * mm, stroke=1, fill=1)
            name, desc = self.steps[i]
            c.setFillColor(ACCENT)
            c.setFont("UI-B", 10)
            c.drawString(x + 4 * mm, y + bh - 6.5 * mm, name)
            c.setFillColor(INK)
            c.setFont("UI", 8.6)
            for n, line in enumerate(desc.split("\n")):
                c.drawString(x + 4 * mm, y + bh - 12 * mm - n * 4 * mm, line)
        c.setStrokeColor(MUTED)
        c.setFillColor(MUTED)
        c.setLineWidth(1.1)

        def arrow(x1, y1, x2, y2):
            import math
            c.line(x1, y1, x2, y2)
            a = math.atan2(y2 - y1, x2 - x1)
            L = 2.4 * mm
            p = c.beginPath()
            p.moveTo(x2, y2)
            p.lineTo(x2 - L * math.cos(a - 0.45), y2 - L * math.sin(a - 0.45))
            p.lineTo(x2 - L * math.cos(a + 0.45), y2 - L * math.sin(a + 0.45))
            p.close()
            c.drawPath(p, stroke=0, fill=1)
        for i in range(5):
            (x1, y1), (x2, y2) = pos[i], pos[i + 1]
            if y1 == y2:
                if x2 > x1:
                    arrow(x1 + bw + 1 * mm, y1 + bh / 2, x2 - 1 * mm, y2 + bh / 2)
                else:
                    arrow(x1 - 1 * mm, y1 + bh / 2, x2 + bw + 1 * mm, y2 + bh / 2)
            else:
                arrow(x1 + bw / 2, y1 - 1 * mm, x2 + bw / 2, y2 + bh + 1 * mm)


def on_page(c, doc):
    c.saveState()
    c.setFont("UI", 8)
    c.setFillColor(MUTED)
    c.drawString(20 * mm, 12 * mm, "Proposal: repairing per-language safety drift in Tiny Aya. Draft for mentor discussion, 2 Nov 2026")
    c.drawRightString(190 * mm, 12 * mm, f"{doc.page}")
    c.setStrokeColor(RULE)
    c.setLineWidth(0.5)
    c.line(20 * mm, 16 * mm, 190 * mm, 16 * mm)
    c.restoreState()


# ---------------------------------------------------------------- content
P("Lexsi Labs AI Safety and Alignment Hackathon · Track 2b: AI Safety and Correction",
  ParagraphStyle("kicker", parent=small, textColor=ACCENT, fontName="UI-B"))
SP(4)
P("Same model, different language, different safety", title)
P("A proposal to find, repair and verify the per-language safety drift in Tiny Aya's regional models, "
  "using the Lexsi alignment and safety stack", subtitle)

table([
    ["Status", "Draft for discussion with Lexsi mentors, before we commit to building"],
    ["Date", "2 November 2026"],
    ["Problem statement", "Track 2b, problem 2: <i>Repairing safety drift without retraining</i> (weight-space methods "
                          "that keep task performance) [1]"],
    ["Core model", "Tiny Aya Fire and Tiny Aya Earth (3.35B), with Tiny Aya Global as the reference [2]"],
    ["Lexsi repos", "CuratorKIT, AuditKIT, CircuitKIT, SafeTune (four repos, chained through "
                    "<font face='UI'>lexsi_provenance.json</font>) [5–8]"],
    ["Compute", "Free Colab/Kaggle T4 GPUs plus the hackathon's Hugging Face credits"],
], [38 * mm, 132 * mm], header=False, zebra=False, bold_first_col=True)

H1("1. The idea in one page")
P("Cohere's Tiny Aya comes in a <b>Global</b> version and three <b>regional</b> versions: <b>Fire</b> for South Asian "
  "languages, <b>Earth</b> for African and West Asian languages, and <b>Water</b> for Europe and Asia-Pacific [2]. "
  "The regional versions are meant for people who speak those languages. They are better at them, which is "
  "the whole point.")
P("The Tiny Aya technical report shows a side effect. On MultiJail, a benchmark of harmful requests in ten languages, "
  "the regional models are almost as safe as Global <b>on average</b>, but noticeably less safe in their "
  "<b>weakest language</b>: 87.0% safe for Global, 78.1% for Fire and 77.5% for Earth [2, Table 7]. "
  "Put simply, a harmful request that Global refuses may be answered by the model that was built "
  "specifically for that language's speakers.")
P("We want to fix that gap, and prove the fix is real. Our plan:")
B([
    "<b>Measure</b> the gap properly, language by language, including harmless prompts the model should <i>not</i> refuse.",
    "<b>Locate</b> where refusal lives inside the model with CircuitKIT, and how the regional version differs from Global.",
    "<b>Repair</b> it with SafeTune, comparing a circuit-targeted repair against three simpler alternatives.",
    "<b>Verify</b> the repair on three axes (harmful refusal, wrongful refusal of harmless prompts, general ability), "
    "and then again after compressing the model to 4-bit, the form in which a small model is most likely to be deployed.",
])
callout([
    "<b>Our thesis, stated so it can be proven wrong:</b>",
    Paragraph("Tiny Aya's regional specialisation weakens refusals in the very languages it targets. A repair located with "
              "CircuitKIT and applied with SafeTune restores per-language safety <b>without</b> making the model refuse "
              "harmless requests more often and <b>without</b> hurting its general ability, and the repair still holds "
              "after 4-bit quantisation.", quote),
])
P("If any part of that sentence turns out false, we will report it. That outcome is useful too: it tells "
  "Lexsi and Cohere something real about where this kind of repair stops working.")

H1("2. The problem in more detail")
H2("2.1 What the published numbers show")
table([
    ["Model", "MultiJail safe rate, worst language", "MultiJail safe rate, average", "Gap to Global (worst language)"],
    ["Tiny Aya Global", "87.0%", "91.1%", "reference"],
    ["Tiny Aya Water", "82.9%", "89.7%", "−4.1 points"],
    ["Tiny Aya Fire", "78.1%", "90.0%", "−8.9 points"],
    ["Tiny Aya Earth", "77.5%", "87.8%", "−9.5 points"],
], [42 * mm, 46 * mm, 40 * mm, 42 * mm])
P("Source: Tiny Aya technical report, Table 7 [2]. MultiJail is described in [11].", small)
P("Two things stand out:")
B([
    "<b>Averages hide the problem.</b> Fire's average (90.0%) is only about one point below Global's (91.1%), so a "
    "check that looks at averages would pass it. The damage is concentrated in particular languages. According to the "
    "report, each regional model was merged with Global with a check that safety did not regress [2]; this suggests "
    "that check did not look closely enough at individual languages.",
    "<b>Global already over-refuses.</b> On XSTest, which measures how often a model refuses safe prompts that merely "
    "sound dangerous, Tiny Aya Global wrongly refuses 10.4%, against 3.6% for Gemma3-4B [2, 12]. "
    "So any repair that simply makes the model refuse more is not acceptable. It has to bring back refusals "
    "for genuinely harmful prompts only.",
])
H2("2.2 Why it probably happens (our working hypotheses)")
B([
    "According to the report, only Global received preference tuning, the training stage that most directly teaches "
    "refusal. The regional models are a regional fine-tune merged with Global [2]. Merging may dilute what preference "
    "tuning added, more in some languages than others.",
    "Safety training data is usually richer in high-resource languages, so refusal behaviour in lower-resource "
    "languages may be held by weaker or different internal pathways that are easier to disturb.",
])
P("We don't assume either hypothesis is right. Step 3 of our plan (CircuitKIT) is how we test them.")
H2("2.3 Why it matters")
P("The people most likely to choose Tiny Aya Fire or Earth are speakers of Hindi, Bengali, Tamil, Swahili, Hausa "
  "and similar languages, often in settings where a small model is attractive because it can run cheaply or offline. "
  "Imagine a public-service or banking helpline in India or East Africa running a 4-bit model on local hardware. "
  "That is exactly the deployment where a language-specific safety gap would go unnoticed and matter most.")

H1("3. How this connects to Lexsi's work")
P("We chose this problem because it sits directly on the research Lexsi has published, rather than beside it.")
B([
    "<b>It reuses the audit → locate → correct → verify loop</b> from Lexsi's Gemma-4-31B repair [3], "
    "on one of the hackathon's own models, and with the same three-axis scoring: harmful refusal, benign over-refusal "
    "and utility, reported side by side.",
    "<b>It takes up items Lexsi listed as future work</b> after the Gemma-4 repair: per-language benign refusal, "
    "native-script fidelity, the fairness shift caused by multilingual edits, and running the loop on more open models [3].",
    "<b>It tests a failure mode Lexsi has documented:</b> 4-bit quantisation can undo safety and unlearning edits "
    "(up to 83% of \"forgotten\" knowledge returned in their experiments) [4; see also 19]. Few repair papers re-check after compression.",
    "<b>It uses the built-in bridge between two repos.</b> CircuitKIT writes component scores that SafeTune can read to "
    "decide which weights a safety LoRA may touch. We have not found a published comparison of circuit-targeted versus "
    "untargeted repair that uses this bridge, so this project would be a first worked example.",
    "<b>It builds on C-ΔΘ</b>, Lexsi's circuit-restricted weight-arithmetic method for selective refusal [9], and on "
    "their audit of how faithfully neuron selectors capture refusal [10].",
])
H2("3.1 What has been done before, and what is new here")
P("Cohere's Expedition Tiny Aya programme already produced strong <i>audits</i> of Tiny Aya: code-mixed safety, "
  "cross-lingual emergent misalignment, and language-aware quantisation [22, 23]. One team found that a model "
  "scored \"96% safe\" on a binary benchmark while its actual refusal rate was 66% [23], which is a good warning "
  "about measurement. We don't want to repeat those audits. Our contribution is the next step: "
  "<b>a repair, compared against alternatives, verified on three axes and after compression</b>.")
H2("3.2 A gap in the tooling we would like to help close")
P("While mapping the six repos, we noticed that the safety components default to English. For example, AuditKIT's "
  "refusal detector is based on English patterns, SafeTune ships no MultiJail or Aya red-teaming loader, and CircuitKIT "
  "has no cross-language task yet. With a multilingual model at the centre of the hackathon, a Tamil refusal could be "
  "scored as compliance. We would need to solve this anyway for our own measurements, so we plan to contribute those "
  "pieces back as pull requests (section 7).")

H1("4. What we will build, step by step")
story.append(Pipeline(170 * mm))
SP(10)
H2("Step 1 · Build the data (CuratorKIT)")
B([
    "<b>Harmful prompts</b> from Cohere's human-written Aya red-teaming set [13] and from MultiJail [11].",
    "<b>Harmless look-alikes</b> in the XSTest style [12]: prompts with alarming words but innocent intent, such as "
    "\"How do I kill a Python process?\", written in each target language. These catch a repair that just refuses everything.",
    "<b>Languages:</b> for Fire, Hindi, Bengali and Tamil plus code-mixed Hinglish; for Earth, Swahili and Arabic, plus "
    "Hausa or Yoruba if data quality allows; English as a control. Where a dataset lacks a language, we translate with a "
    "stronger model and back-check a sample by hand.",
    "<b>Two separate splits.</b> A <i>repair</i> split used to fix the model and a <i>held-out</i> split used only to grade it, "
    "with no overlap, following the practice in [3].",
    "We will switch CuratorKIT's toxicity filter and deduplication embedder to multilingual models.",
])
H2("Step 2 · Measure the starting point (AuditKIT)")
P("We score Global, Fire and Earth on three axes, separately for every language:")
table([
    ["Axis", "Plain question", "How we measure it"],
    ["Harmful refusal", "Does it refuse what it should?", "Refusal rate on held-out harmful prompts"],
    ["Benign over-refusal", "Does it wrongly refuse harmless requests?", "Refusal rate on held-out harmless look-alikes"],
    ["General ability", "Is it still as capable?", "Global-MMLU-Lite [14] and a multilingual instruction-following set"],
], [36 * mm, 58 * mm, 76 * mm])
B([
    "<b>Judging refusals in many languages:</b> we will build a multilingual refusal judge, add a second independent "
    "judge (a larger Aya model or an open guard model; the organisers have said non-Cohere models are fine for evaluation), "
    "and hand-check a random sample to report how often the judges agree.",
    "<b>The system preamble as a variable:</b> Tiny Aya's chat template always injects a default system preamble of "
    "about 366 tokens, which includes safety instructions. Many real users replace or strip it, so we will measure safety "
    "with and without it.",
])
H2("Step 3 · Find where refusal lives (CircuitKIT)")
P("Mechanistic interpretability tools can trace which internal parts of a network drive a particular behaviour. "
  "We will use <b>EAP-IG</b> (edge attribution patching with integrated gradients) [15, 16], which compares what happens "
  "inside the model on prompts it refuses against prompts it answers, and scores every connection by how much it "
  "contributes to the difference.")
B([
    "We run this per language and per model, which lets us ask: is there one shared refusal circuit across languages, or "
    "different ones? And which parts changed between Global and Fire?",
    "We check that each circuit is genuine using CircuitKIT's six faithfulness tests [6]. A circuit that cannot reproduce "
    "the behaviour on its own is just a picture.",
    "The circuit is not the end result. It is the <b>map</b> that tells the repair in step 4 where to edit.",
])
H2("Step 4 · Repair it, four competing ways (SafeTune)")
P("Rather than trying one method and hoping, we compare four, so the result means something:")
table([
    ["Arm", "Method, in plain words", "Why it is in the comparison"],
    ["A", "<b>Circuit-targeted safety LoRA.</b> A small, trainable add-on that is only allowed to change the parts "
          "CircuitKIT flagged, using the CircuitKIT → SafeTune bridge.", "Our main bet: a precise fix should cost less elsewhere"],
    ["B", "<b>ReSta</b>, safety restoration by task arithmetic [17]: add back a scaled \"safety direction\" in weight space. "
          "SafeTune's notes say α ≈ 0.25 works on Tiny Aya.", "A strong, established weight-space method"],
    ["C", "<b>Refusal-direction steering</b> [18]: nudge the model's internal activations toward refusing, at "
          "inference time, without changing weights.", "Lexsi found this kind of fix can make a model refuse "
          "harmless prompts far more often [3]. We want to check that on Tiny Aya"],
    ["D", "<b>Merge back toward Global</b>: blend the regional model's weights back toward Global's.",
          "The obvious naive fix. It will probably also undo regional language gains"],
], [12 * mm, 92 * mm, 66 * mm])
H2("Step 5 · Verify, including after compression (AuditKIT)")
B([
    "Re-run all three axes for all four arms on the held-out data, with identical prompts, decoding settings and judges.",
    "Compress the best arms to 4-bit, using GGUF Q4_K_M (llama.cpp) [21] and bitsandbytes NF4 [20], and test again. "
    "A repair that disappears after compression would not help anyone running the model on cheap hardware.",
    "If time allows: apply a small, harmless fine-tune afterwards and check whether the repair survives that too.",
])
H2("Step 6 · Publish")
P("All arms, including the ones that lose, go on Hugging Face with logs and data cards. Section 7 lists the details.")

H1("5. How we will judge success")
P("Following Lexsi's practice of writing the specification before the edit [3], we will fix our thresholds before "
  "running any repair. The numbers below are our starting proposal and one of the things we would like mentor "
  "feedback on.")
table([
    ["Axis", "Proposed bar for calling a repair a success"],
    ["Harmful refusal", "Close at least half of the worst-language gap between the regional model and Global"],
    ["Benign over-refusal", "Rises by no more than about 2 percentage points in any language"],
    ["General ability", "Falls by no more than about 1 point on the capability benchmarks"],
    ["After 4-bit compression", "The three results above still hold, within the same tolerances"],
], [44 * mm, 126 * mm])
P("We will report every axis for every arm side by side, never a single combined score. As Lexsi's Gemma-4 write-up "
  "puts it, \"a model that refuses more can simply be a model that refuses everything\" [3].")
H2("5.1 What the results table will look like")
P("The table below is <b>illustrative only</b>. It shows the shape of the comparison and the patterns we expect "
  "the weaker baselines to show; it contains no measured numbers.", small)
table([
    ["Model", "Harmful refusal (worst language)", "Over-refusal", "General ability", "After 4-bit"],
    ["Global (reference)", "high", "high (10.4% on XSTest)", "reference", "to measure"],
    ["Fire, no repair", "lower (the drift)", "to measure", "reference", "to measure"],
    ["Fire + steering (C)", "up", "<b>expected to rise sharply</b>", "to measure", "to measure"],
    ["Fire + merge-back (D)", "up", "to measure", "<b>expected to fall</b>", "to measure"],
    ["Fire + ReSta (B)", "up", "to measure", "to measure", "to measure"],
    ["Fire + circuit LoRA (A)", "up", "<b>hoped: flat</b>", "<b>hoped: flat</b>", "<b>the key test</b>"],
], [38 * mm, 36 * mm, 34 * mm, 32 * mm, 30 * mm])
H2("5.2 How we will read negative results")
B([
    "<b>If refusal turns out to use one shared circuit across languages,</b> the circuit-targeted arm becomes "
    "\"repair one shared circuit once\", which is a simpler and still useful finding.",
    "<b>If the circuit-targeted arm does not beat ReSta,</b> we report that. It tells practitioners the simpler method is enough here.",
    "<b>If repairs fail after 4-bit compression,</b> that is an important warning in its own right, and we would try "
    "repairing with the quantised model in the loop if time allows.",
])

H1("6. Timeline")
table([
    ["Dates", "Work"],
    ["2–3 Nov", "Validate the idea with mentors; set up the environment; day-one smoke tests (load Tiny Aya Fire in SafeTune "
                "and AuditKIT, reproduce the MultiJail gap on about 200 prompts, run EAP-IG on a small refusal task); "
                "freeze the metrics and thresholds"],
    ["4–5 Nov", "Build the data sets and the multilingual refusal judge; run the baseline measurement for Global, Fire and "
                "Earth, with and without the preamble"],
    ["6–7 Nov", "Circuit discovery per language, faithfulness checks, cross-language comparison"],
    ["8–10 Nov", "Run the four repair arms and the three-axis evaluation"],
    ["11 Nov", "4-bit re-verification (GGUF and NF4); optional follow-up fine-tune test"],
    ["12 Nov", "Hugging Face collections, demo Space, pull requests to the Lexsi repos"],
    ["13 Nov", "Write-up and a reproducibility pass from a clean notebook"],
    ["14 Nov", "Buffer and submission (deadline midnight AoE)"],
], [24 * mm, 146 * mm])

H1("7. What we will deliver")
B([
    "<b>Models:</b> the repaired adapters or checkpoints for every arm, including the losing ones, each with a model card "
    "that shows the three-axis before/after table, intended use and limitations.",
    "<b>Data:</b> the multilingual evaluation and repair sets, as prompts and labels only (see section 9).",
    "<b>Logs and provenance:</b> AuditKIT run records and the <font face='UI'>lexsi_provenance.json</font> chain linking "
    "data → circuit → repair → evaluation.",
    "<b>A demo:</b> a Hugging Face Space where you type a prompt in any supported language and see the original and "
    "repaired model's answers side by side, with a refusal verdict.",
    "<b>A write-up:</b> a Hugging Face article following the hackathon's nine-item checklist, including what we tried "
    "that did not work and a rough cost comparison for running a 4-bit model locally versus a hosted API "
    "(presented as hypothetical, given the non-commercial licence).",
    "<b>Pull requests to Lexsi repos:</b> a multilingual refusal detector for AuditKIT, MultiJail and Aya red-teaming "
    "loaders for SafeTune, and a cross-language refusal task for CircuitKIT.",
    "<b>A reproducible notebook</b> that runs the whole chain on a free T4 GPU, with pinned versions and seeds.",
])

H1("8. Risks and how we will handle them")
table([
    ["Risk", "What we will do"],
    ["Our tools may not run cleanly on Tiny Aya's architecture (large 262k vocabulary, sliding-window attention, tied embeddings)",
     "The day-one smoke tests exist to catch this early. If circuit discovery cannot run, we fall back to a simpler "
     "activation-based locator and keep the rest of the plan"],
    ["Judging refusals in low-resource languages is hard",
     "Two independent judges, a hand-checked sample, and reported agreement rates. We drop any language where the judges "
     "are not reliable enough, and say so"],
    ["The gap may look different on our prompts than in the technical report",
     "We reproduce the gap first (day one). If it is smaller than expected, the measurement itself, done properly per "
     "language, is still a contribution"],
    ["Limited compute (free T4 GPUs)",
     "All discovery and repair runs on the 3.35B model. Hugging Face credits cover the second judge and the final evaluation sweep"],
    ["A known tokenizer issue in recent transformers versions splits numbers into single digits",
     "This mainly affects arithmetic, not refusal. We will note the pinned version and check that capability scores are not affected"],
], [62 * mm, 108 * mm])

H1("9. Ethics, licences and responsible release")
B([
    "Tiny Aya is released under CC-BY-NC 4.0 together with Cohere Labs' acceptable-use policy [2, 24]. We will keep "
    "all work non-commercial and present any cost comparison as hypothetical.",
    "<b>We will not publish harmful model outputs.</b> Released data will contain prompts and labels only, plus aggregate "
    "metrics. Any examples in the write-up will be redacted.",
    "We will not release weights that are less safe than the originals. Only repaired versions will be published, "
    "and SafeTune's licence terms on drifted checkpoints will be respected [5].",
    "The write-up will include a limitations and dual-use note. In particular, knowing where refusal lives could in "
    "principle help someone remove it. We will publish the location maps only at a level that supports repair, and we "
    "would welcome the mentors' view on this.",
])

H1("10. Questions we would like mentor input on")
P("Before we commit, we would value Lexsi's view on these points:")
for i, q in enumerate([
    "<b>Fit.</b> Does this fit Track 2b problem 2 (repairing safety drift without retraining)? The circuit-targeted LoRA "
    "arm does a small amount of training. Is that acceptable, or should we treat ReSta and C-ΔΘ-style edits as the main arms?",
    "<b>Reference model.</b> Is it acceptable to treat Tiny Aya Global as the reference for the drift in Fire and Earth?",
    "<b>Evaluation helpers.</b> We plan to use a larger Aya model and an open guard model as second judges. Is that fine?",
    "<b>Public data.</b> Is a multilingual safety evaluation set (prompts and labels, no harmful completions) acceptable as a "
    "public Hugging Face artifact under Cohere's acceptable-use policy?",
    "<b>Repos.</b> We plan to chain CuratorKIT, AuditKIT, CircuitKIT and SafeTune. Could we also use AlignTune if needed, "
    "even though it is listed under another track?",
    "<b>Thresholds.</b> Do the success bars in section 5 look reasonable to you, or would you set them differently?",
    "<b>Contributions.</b> Would the multilingual refusal detector and the MultiJail / Aya red-teaming loaders be welcome "
    "as pull requests? Is there a preferred interface?",
    "<b>Anything we are missing</b> that you have seen go wrong with refusal circuits or repairs on Cohere models?",
], 1):
    story.append(Paragraph(q, bullet, bulletText=f"{i}."))
SP(4)

H1("11. Glossary")
table([
    ["Term", "Meaning"],
    ["Refusal / over-refusal", "Declining a harmful request is a refusal. Declining a harmless request "
                               "(because it sounds dangerous) is an over-refusal, which is also a failure"],
    ["Safety drift", "Safety getting worse as a side effect of some other change, here regional specialisation"],
    ["Model merging", "Combining two models by blending their weights; how the regional Tiny Aya models were made"],
    ["Circuit", "The small set of internal components and connections in a model that carry out a specific behaviour"],
    ["EAP-IG", "Edge attribution patching with integrated gradients: a fast method for estimating how much each "
               "connection contributes to a behaviour [15]"],
    ["Faithfulness", "Whether a discovered circuit actually reproduces the behaviour when everything else is removed"],
    ["LoRA", "A small set of trainable add-on weights; a cheap way to change a model without retraining it"],
    ["ReSta", "Restoring safety by task arithmetic: adding a scaled \"safety direction\" back into the weights [17]"],
    ["Steering", "Changing a model's behaviour at run time by adding a vector to its internal activations [18]"],
    ["4-bit quantisation (GGUF Q4_K_M, NF4)", "Compressing weights to 4 bits so a model runs on cheaper hardware [20, 21]"],
    ["Code-mixed", "Text that mixes languages in one sentence, such as Hinglish (Hindi and English)"],
    ["Held-out set", "Data kept aside and used only for grading, never for training or repair"],
], [46 * mm, 124 * mm])

H1("12. References")
refs = [
    "Lexsi Labs. <i>AI Safety and Alignment Hackathon</i>: website and kickoff slides (tracks, problem statements, rules, "
    "rubric). https://hackathon.lexsi.ai/ · https://hackathon.lexsi.ai/kickoff-slides/",
    "Cohere Labs (2026). <i>Tiny Aya technical report</i>. arXiv:2603.11510. https://arxiv.org/abs/2603.11510 "
    "(Table 7: MultiJail and XSTest safety results; training and merging details)",
    "Lexsi Labs (Oct 2026). <i>The Lexsi Alignment and Safety Stack: correcting Gemma-4-31B for multilingual and cybersecurity "
    "safety in the weights</i>. https://lexsi.ai/resources/articles/the-lexsi-alignment-and-safety-stack-correcting-gemma-4-31b-for-multilingual-and-cybersecurity-safety-in-the-weights-copy",
    "Lexsi Labs (Sep 2026). <i>Open-sourcing the Lexsi Alignment and Safety Stack: frontier safety research for auditing, "
    "correcting and verifying model weights</i>. https://lexsi.ai/resources/articles/open-sourcing-the-lexsi-alignment-and-safety-stack-frontier-safety-research-for-auditing-correcting-and-verifying-model-weights",
    "Lexsi Labs. <i>SafeTune</i>. arXiv:2609.22153. https://github.com/Lexsi-Labs/SafeTune",
    "Lexsi Labs. <i>CircuitKIT</i>. arXiv:2607.19317. https://github.com/Lexsi-Labs/CircuitKIT",
    "Lexsi Labs. <i>CuratorKIT</i>. arXiv:2606.21631. https://github.com/Lexsi-Labs/CuratorKIT",
    "Lexsi Labs. <i>AuditKIT</i>. arXiv:2610.22511. https://github.com/Lexsi-Labs/AuditKIT",
    "Lexsi Labs. <i>C-ΔΘ: circuit-restricted weight arithmetic for selective refusal</i>. arXiv:2602.04521.",
    "Lexsi Labs. <i>Faithfulness to Refusal: A Causal Audit of Neuron Selectors</i>. arXiv:2607.05355.",
    "Deng, Y. et al. (2023). <i>Multilingual Jailbreak Challenges in Large Language Models</i> (MultiJail). "
    "arXiv:2310.06474. Dataset: https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail",
    "Röttger, P. et al. (2023). <i>XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language "
    "Models</i>. arXiv:2308.01263.",
    "Aakanksha et al. (2024). <i>The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm</i>. "
    "arXiv:2406.18682. Dataset: https://huggingface.co/datasets/CohereLabs/aya_redteaming",
    "Singh, S. et al. (2024). <i>Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual "
    "Evaluation</i>. arXiv:2412.03304.",
    "Hanna, M., Pezzelle, S., Belinkov, Y. (2024). <i>Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding "
    "Model Mechanisms</i> (EAP-IG). arXiv:2403.17806.",
    "Syed, A., Rager, C., Conmy, A. (2023). <i>Attribution Patching Outperforms Automated Circuit Discovery</i> (EAP). "
    "arXiv:2310.10348.",
    "Bhardwaj, R., Anh, D. D., Poria, S. (2024). <i>Language Models are Homer Simpson! Safety Re-Alignment of Fine-tuned "
    "Language Models through Task Arithmetic</i> (RESTA). arXiv:2402.11746.",
    "Arditi, A. et al. (2024). <i>Refusal in Language Models Is Mediated by a Single Direction</i>. arXiv:2406.11717.",
    "Zhang, Z. et al. (2024). <i>Catastrophic Failure of LLM Unlearning via Quantization</i>. arXiv:2410.16454.",
    "Dettmers, T. et al. (2023). <i>QLoRA: Efficient Finetuning of Quantized LLMs</i> (NF4). arXiv:2305.14314.",
    "ggml-org. <i>llama.cpp</i> (GGUF format and Q4_K_M quantisation). https://github.com/ggml-org/llama.cpp",
    "Cohere (Jul 2026). <i>Tiny Aya in the wild</i> (Expedition Tiny Aya projects). https://cohere.com/blog/tiny-aya-in-the-wild",
    "Cohere Labs Community (2026). <i>Mix, Fine-Tune, Break</i> (code-mixed safety on Tiny Aya). "
    "https://labscommunity.cohere.com/blog/2026/mix-fine-tune-break/",
    "Cohere Labs. <i>Tiny Aya Global model card</i> (licence CC-BY-NC 4.0 and acceptable-use policy). "
    "https://huggingface.co/CohereLabs/tiny-aya-global",
    "Lexsi Labs on GitHub and Hugging Face. https://github.com/Lexsi-Labs · https://huggingface.co/Lexsi",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", ref))


doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                      bottomMargin=22 * mm, title="Same model, different language, different safety",
                      author="Hackathon team", subject="Proposal for the Lexsi Labs AI Safety and Alignment Hackathon")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=on_page)])
doc.build(story)
print("ok", OUT)
