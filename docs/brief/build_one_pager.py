"""One page for mentors on what the project builds (not its results). Writes docs/brief/what_we_are_building.pdf.

    .venv/Scripts/python.exe docs/brief/build_one_pager.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).with_name('what_we_are_building.pdf')
INK, ACCENT, FILL, MUTED = colors.HexColor('#1B2A3A'), colors.HexColor('#1F5F8B'), colors.HexColor('#EEF4F9'), colors.HexColor('#5B6B7A')

title = ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=17, leading=21, textColor=INK)
subtitle = ParagraphStyle('subtitle', fontName='Helvetica', fontSize=9, leading=12, textColor=MUTED)
heading = ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=10.5, leading=14, textColor=ACCENT,
                         spaceBefore=7, spaceAfter=2)
body = ParagraphStyle('body', fontName='Helvetica', fontSize=9, leading=12.2, textColor=INK, alignment=TA_LEFT)
bullet = ParagraphStyle('bullet', parent=body, leftIndent=10, bulletIndent=0, spaceAfter=1)
step_head = ParagraphStyle('step_head', fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=ACCENT)
step_body = ParagraphStyle('step_body', parent=body, fontSize=8.4, leading=11)


def bullets(items):
    return [Paragraph(item, bullet, bulletText='•') for item in items]


STEPS = [
    ('1  Measure',
     'Seven languages: Arabic, Bengali, English, Javanese, Korean, Swahili and Thai. Harmful requests, and harmless '
     'look-alikes to catch over-refusal, each with and without a safety preamble, on held-out prompts. Two independent '
     'judges label every answer for content safety and refusal. A multilingual knowledge test tracks capability.'),
    ('2  Localise',
     'CircuitKIT maps where refusal is computed inside Tiny Aya Global and Tiny Aya Earth, language by language, so '
     'an edit can be aimed at those parts of the model.'),
    ('3  Repair',
     'Five edits with no retraining, using SafeTune: ReSta, an edit confined to the circuit, merging back toward '
     'Global, a merge confined to the circuit, and activation steering. The same edit on random layers is the control '
     'that tests whether the circuit map helps at all.'),
    ('4  Check',
     'Pass conditions fixed in advance. Close at least half the safety gap. Raise benign over-refusal by at most 2 '
     'points and lower capability by at most 1 point in any language. Still hold after 4-bit quantisation, in both '
     'GGUF Q4_K_M and bitsandbytes NF4.'),
]


def build() -> None:
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.3 * cm, title='What we are building',
                            author='Aditya Arnav and Pallab Das', subject='Lexsi Labs hackathon, Track 2b')
    width = A4[0] - doc.leftMargin - doc.rightMargin
    story = [
        Paragraph('Same model, different language, different safety', title),
        Paragraph('What we are building · Lexsi Labs AI Safety &amp; Alignment Hackathon, Track 2b '
                  '· Aditya Arnav and Pallab Das', subtitle),
        Spacer(1, 6),
        Paragraph('The problem', heading),
        Paragraph(
            'Tiny Aya is a family of small (3.35B) open-weights models from Cohere Labs: Global, plus regional variants '
            'tuned for parts of the world. Its technical report shows the Earth variant, aimed at Africa and West Asia, '
            '9.5 points less safe than Global in its worst language, while the average across languages falls only 3.3 '
            'points. The report does not say which language. A team running a small model on its own hardware, such as '
            'a public-service helpline, needs safety numbers for the languages its users speak, measured on the 4-bit '
            'file it will actually deploy.', body),
        Paragraph('What we are building', heading),
        Paragraph(
            'An open, reproducible pipeline that finds where a regional model’s safety drops language by language, '
            'tries to repair it without retraining, and checks that the repair costs nothing elsewhere and survives '
            'quantisation. It is built on the Lexsi Labs kits (CircuitKIT, SafeTune, AuditKIT, CuratorKIT), with Tiny '
            'Aya Earth against Tiny Aya Global as the test case.', body),
        Spacer(1, 6),
    ]
    cells = [[Paragraph(head, step_head), Spacer(1, 3), Paragraph(text, step_body)] for head, text in STEPS]
    gap = 0.25 * cm
    table = Table([cells], colWidths=[(width - 3 * gap) / 4] * 4, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), FILL), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBEFORE', (1, 0), (-1, 0), gap, colors.white), ('LINEABOVE', (0, 0), (-1, 0), 2, ACCENT),
    ]))
    story.append(table)
    story += [
        Paragraph('The questions it answers', heading),
        *bullets([
            'Is the safety gap real on held-out prompts at full precision, and in which languages?',
            'Is it unsafe content, or the model breaking down in that language? We keep the two apart throughout.',
            'Can the Lexsi stack close the gap without retraining, and without raising over-refusal or losing '
            'capability?',
            'Does knowing where refusal lives make the repair better than editing random layers?',
            'Does the result survive 4-bit quantisation, the form small models are actually deployed in?',
        ]),
        Paragraph('How we keep it honest', heading),
        *bullets([
            'A frozen specification: thresholds and tests fixed before any repair ran, and every later change logged '
            'as a dated amendment.',
            'Paired 95% confidence intervals, with McNemar tests corrected for all 14 language and preamble '
            'comparisons. Prompts for choosing repairs are kept apart from prompts for measuring them.',
            'Judges checked three ways: two independent local judges, agreement with the larger pre-registered judges '
            'on a stratified sample, and blind re-reads of sampled answers. Checks by native speakers are planned.',
            'Every run is recorded in a hash-chained ledger. Runs moved to rented GPUs must match the laptop run file '
            'by file, by SHA-256, before they start.',
            'Harmful prompts and the models’ answers to them are never published.',
        ]),
        Paragraph('What we will hand over', heading),
        *bullets([
            'The pipeline as open source, rerunnable end to end on one consumer GPU.',
            'A write-up in which every number traces back to the file it came from, including the results that go '
            'against our starting thesis.',
            'Proposed fixes for the Lexsi kits, found while building on them.',
        ]),
        Paragraph('Where we are', heading),
        Paragraph(
            'Measurement, the circuit maps and all five repairs are done at full precision. The 4-bit checks are '
            'running now, and the write-up is due on 14 November, anywhere on Earth.', body),
    ]
    doc.build(story)
    if doc.page != 1:
        raise SystemExit(f'The page overflowed onto {doc.page} pages; shorten it')
    print(OUT)


if __name__ == '__main__':
    build()
