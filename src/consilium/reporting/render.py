"""PDF and Word renderings of a ``ReportDocument`` (no logic of their own:
everything shown comes from the document, including its state banner)."""

from __future__ import annotations

import io
from datetime import datetime

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from fpdf import FPDF

from consilium.reporting.document import ReportDocument

__all__ = ["render_docx", "render_pdf"]

OVERLAY_CAPTION = "Optic disc (cyan) and cup (magenta) outlines from the stored segmentation."

# banner colours per state (RGB)
_TONE: dict[str, tuple[int, int, int]] = {
    "approved": (22, 128, 61),
    "draft": (180, 110, 0),
    "unverified": (190, 30, 30),
    "rejected": (110, 110, 110),
}

# Core PDF fonts are Latin-1; map common typography and drop the rest.
_ASCII = str.maketrans(
    {
        chr(code): repl
        for code, repl in (
            (0x2014, "-"),  # em dash
            (0x2013, "-"),  # en dash
            (0x2018, "'"),
            (0x2019, "'"),
            (0x201C, '"'),
            (0x201D, '"'),
            (0x2026, "..."),
            (0x2265, ">="),
            (0x2264, "<="),
            (0x2192, "->"),
            (0x2022, "-"),
            (0x00D7, "x"),
            (0x00B7, "-"),
        )
    }
)


def _latin1(text: str) -> str:
    return text.translate(_ASCII).encode("latin-1", "replace").decode("latin-1")


def _when(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "-"


def _case_rows(doc: ReportDocument) -> list[tuple[str, str]]:
    rows = [
        ("Case ID", doc.case_id),
        ("Eye", doc.eye),
        ("Submitted", _when(doc.created_at)),
        ("Report generated", _when(doc.generated_at)),
        ("AI draft revision", str(doc.draft_revision)),
    ]
    if doc.approved_by:
        rows += [
            ("Approved by", doc.approved_by),
            ("Approved at", _when(doc.approved_at)),
            ("Report version", str(doc.version)),
        ]
        if doc.reason_codes:
            rows.append(("Edit reason codes", ", ".join(doc.reason_codes)))
    return rows


# ------------------------------------------------------------------ PDF


class _Pdf(FPDF):
    footer_text = ""

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(110, 110, 110)
        self.multi_cell(
            0, 3.6, _latin1(f"{self.footer_text}   Page {self.page_no()}/{{nb}}"), align="C"
        )


def render_pdf(doc: ReportDocument, overlay_png: bytes | None) -> bytes:
    pdf = _Pdf(format="A4")
    pdf.footer_text = doc.footer
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(16, 14, 16)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    def heading(text: str) -> None:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11.5)
        pdf.set_text_color(20, 40, 70)
        pdf.cell(0, 7, _latin1(text), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 210, 225)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + width, pdf.get_y())
        pdf.ln(1.5)
        pdf.set_text_color(25, 25, 25)

    def body(text: str, size: float = 9.5, style: str = "") -> None:
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, 4.8, _latin1(text), new_x="LMARGIN", new_y="NEXT")

    def bullet(text: str, size: float = 9.5) -> None:
        pdf.set_font("Helvetica", "", size)
        pdf.cell(5, 4.8, "-")
        pdf.multi_cell(width - 5, 4.8, _latin1(text), new_x="LMARGIN", new_y="NEXT")

    def table(rows: list[tuple[str, str]], left: float = 52) -> None:
        for label, value in rows:
            pdf.set_font("Helvetica", "B", 9)
            y = pdf.get_y()
            pdf.multi_cell(left, 5, _latin1(label), new_x="RIGHT", new_y="TOP")
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(width - left, 5, _latin1(value), new_x="LMARGIN", new_y="NEXT")
            pdf.set_y(max(pdf.get_y(), y + 5))

    # title
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(15, 30, 60)
    pdf.cell(0, 9, "CONSILIUM", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(70, 80, 100)
    pdf.cell(0, 6, _latin1(doc.title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # state banner
    r, g, b = _TONE[doc.state]
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, _latin1("  " + doc.state_label), fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.multi_cell(
        0, 4.6, _latin1("  " + doc.state_detail), fill=True, new_x="LMARGIN", new_y="NEXT"
    )
    pdf.set_text_color(25, 25, 25)

    heading("Case")
    table(_case_rows(doc))

    heading("Image findings (SwinV2 + SegFormer)")
    top = pdf.get_y()
    img_w = 62.0
    if overlay_png:
        pdf.image(io.BytesIO(overlay_png), x=pdf.l_margin, y=top, w=img_w)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_xy(pdf.l_margin, top + img_w + 1)
        pdf.multi_cell(img_w, 3.4, OVERLAY_CAPTION)
        bottom = pdf.get_y()
        pdf.set_xy(pdf.l_margin + img_w + 5, top)
        left = pdf.l_margin + img_w + 5
    else:
        bottom = top
        left = pdf.l_margin
    for row in doc.image_findings:
        pdf.set_x(left)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(58, 5, _latin1(row.label))
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, _latin1(row.value), new_x="LMARGIN", new_y="NEXT")
        if row.note:
            pdf.set_x(left)
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(
                pdf.w - pdf.r_margin - left, 3.8, _latin1(row.note), new_x="LMARGIN", new_y="NEXT"
            )
            pdf.set_text_color(25, 25, 25)
    pdf.set_y(max(pdf.get_y(), bottom) + 1)

    if doc.note_facts:
        heading("Clinical note (de-identified, as extracted)")
        table([(r.label, r.value) for r in doc.note_facts])

    heading("Impression")
    body(doc.impression, 10.5, "B")

    heading("Findings")
    for f in doc.findings:
        body(f"{f.claim_id}. {f.text}", 9.5)
        evidence = "; ".join(f"{e.label}: {e.value}" for e in f.evidence)
        pdf.set_text_color(90, 90, 90)
        body(f"     Evidence: {evidence}   (from: {', '.join(f.source_roles)})", 8)
        pdf.set_text_color(190, 30, 30)
        for issue in f.flagged:
            body(f"     OPEN ISSUE - {issue}", 8, "B")
        pdf.set_text_color(25, 25, 25)

    if doc.open_issues:
        heading("Open issues (whole draft)")
        pdf.set_text_color(190, 30, 30)
        for issue in doc.open_issues:
            bullet(issue)
        pdf.set_text_color(25, 25, 25)

    if doc.disagreements:
        heading("Preserved disagreements between specialists")
        for d in doc.disagreements:
            body(d.topic, 9.5, "B")
            for role, text in d.positions:
                bullet(f"{role}: {text}", 9)
            body(f"Resolution: {d.resolution}", 8.5, "I")

    heading("Recommendations (for clinician judgement)")
    if doc.recommendations:
        for rec in doc.recommendations:
            bullet(f"{rec.text}  (triggered by: {', '.join(rec.triggered_by)})")
    else:
        body("None.", 9.5, "I")

    heading("Limitations")
    for item in doc.limitations:
        bullet(item)

    heading("Model information")
    table([(r.label, r.value) for r in doc.model_info], left=44)

    return bytes(pdf.output())


# ------------------------------------------------------------------ Word


def _table(document: object, rows: list[tuple[str, str]]) -> None:
    table = document.add_table(rows=0, cols=2)  # type: ignore[attr-defined]
    table.style = "Light Grid Accent 1"
    for label, value in rows:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = value
        for p in cells[0].paragraphs:
            for run in p.runs:
                run.bold = True


def render_docx(doc: ReportDocument, overlay_png: bytes | None) -> bytes:
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    document.add_heading("CONSILIUM", level=0)
    document.add_paragraph(doc.title)

    banner = document.add_paragraph()
    run = banner.add_run(doc.state_label)
    run.bold = True
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(*_TONE[doc.state])
    document.add_paragraph(doc.state_detail).runs[0].italic = True

    document.add_heading("Case", level=1)
    _table(document, _case_rows(doc))

    document.add_heading("Image findings (SwinV2 + SegFormer)", level=1)
    if overlay_png:
        document.add_picture(io.BytesIO(overlay_png), width=Inches(2.8))
        cap = document.add_paragraph(OVERLAY_CAPTION)
        cap.runs[0].italic = True
        cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _table(
        document,
        [(r.label, r.value + (f"  —  {r.note}" if r.note else "")) for r in doc.image_findings],
    )

    if doc.note_facts:
        document.add_heading("Clinical note (de-identified, as extracted)", level=1)
        _table(document, [(r.label, r.value) for r in doc.note_facts])

    document.add_heading("Impression", level=1)
    document.add_paragraph().add_run(doc.impression).bold = True

    document.add_heading("Findings", level=1)
    for f in doc.findings:
        document.add_paragraph(f"{f.claim_id}. {f.text}", style="List Number")
        evidence = "; ".join(f"{e.label}: {e.value}" for e in f.evidence)
        ev = document.add_paragraph(f"Evidence: {evidence}  (from: {', '.join(f.source_roles)})")
        ev.runs[0].font.size = Pt(8.5)
        ev.runs[0].font.color.rgb = RGBColor(90, 90, 90)
        for issue in f.flagged:
            r = document.add_paragraph().add_run(f"OPEN ISSUE — {issue}")
            r.bold = True
            r.font.color.rgb = RGBColor(190, 30, 30)

    if doc.open_issues:
        document.add_heading("Open issues (whole draft)", level=1)
        for issue in doc.open_issues:
            document.add_paragraph(issue, style="List Bullet")

    if doc.disagreements:
        document.add_heading("Preserved disagreements between specialists", level=1)
        for d in doc.disagreements:
            document.add_paragraph().add_run(d.topic).bold = True
            for role, text in d.positions:
                document.add_paragraph(f"{role}: {text}", style="List Bullet")
            document.add_paragraph(f"Resolution: {d.resolution}").runs[0].italic = True

    document.add_heading("Recommendations (for clinician judgement)", level=1)
    for rec in doc.recommendations or ():
        document.add_paragraph(
            f"{rec.text}  (triggered by: {', '.join(rec.triggered_by)})", style="List Bullet"
        )
    if not doc.recommendations:
        document.add_paragraph("None.")

    document.add_heading("Limitations", level=1)
    for item in doc.limitations:
        document.add_paragraph(item, style="List Bullet")

    document.add_heading("Model information", level=1)
    _table(document, [(r.label, r.value) for r in doc.model_info])

    footer = document.sections[0].footer.paragraphs[0]
    footer.text = doc.footer
    footer.runs[0].font.size = Pt(8)
    footer.runs[0].italic = True

    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()
