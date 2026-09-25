"""Build the 44-page, fillable college report from editable Markdown source."""
import json
import re
import shutil
from html import escape
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image, Frame, Flowable
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader

BASE = Path(__file__).resolve().parents[1]
DOC = BASE / "documentation"
DETAILS = json.loads((DOC / "submission_details.json").read_text())
TITLE = "AI-Based Customer Segmentation and Personalized Marketing Recommendation System using Machine Learning"
TEAL, INK = colors.HexColor("#0d9488"), colors.HexColor("#173342")
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.8, leading=16.2, textColor=INK, spaceAfter=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=9, leading=12, spaceAfter=5)
HEAD = ParagraphStyle("head", parent=BODY, fontName="Helvetica-Bold", fontSize=21, leading=26, spaceAfter=16)
SUB = ParagraphStyle("sub", parent=BODY, fontName="Helvetica-Bold", fontSize=13, leading=18, textColor=TEAL, spaceAfter=12)


def markup(text):
    text = escape(text).replace("&quot;", '"')
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


class Diagram(Flowable):
    def __init__(self, kind):
        super().__init__()
        self.width, self.height = 487, 185
        labels = {
            "architecture": ["Administrator", "Login / session gate", "Protected Streamlit UI", "Analytics modules", "SQLite history", "Model / CSV exports"],
            "ml_workflow": ["Validate input", "Compute RFM", "Fit preprocessing", "Fit four algorithms", "Compare valid metrics", "Profiles and artifacts"],
            "data_flow": ["CSV + purchase ledger", "Validated snapshot", "SQLite dataset ID", "Read + train models", "Run ID and results", "Dashboard + downloads"],
            "clustering": ["Select k", "Initialize centers", "Assign nearest center", "Update group means", "Repeat until stable", "Labels + prediction"],
        }
        self.labels = labels[kind]

    def draw(self):
        c = self.canv
        positions = [(0, 115), (171, 115), (342, 115), (342, 30), (171, 30), (0, 30)]
        for i, ((x, y), label) in enumerate(zip(positions, self.labels)):
            c.setFillColor(colors.HexColor("#eef7f5")); c.setStrokeColor(TEAL)
            c.roundRect(x, y, 145, 48, 6, fill=1, stroke=1)
            c.setFillColor(INK); c.setFont("Helvetica-Bold", 9.5)
            c.drawCentredString(x + 72.5, y + 21, label)
            if i < 5:
                if i < 2:
                    c.line(x + 145, y + 24, x + 165, y + 24)
                    c.line(x + 165, y + 24, x + 160, y + 27); c.line(x + 165, y + 24, x + 160, y + 21)
                elif i == 2:
                    c.line(x + 72.5, y, x + 72.5, y - 31)
                    c.line(x + 72.5, y - 31, x + 69.5, y - 26); c.line(x + 72.5, y - 31, x + 75.5, y - 26)
                else:
                    c.line(x, y + 24, x - 20, y + 24)
                    c.line(x - 20, y + 24, x - 15, y + 27); c.line(x - 20, y + 24, x - 15, y + 21)


def table(rows, widths=None):
    count = len(rows[0])
    if widths is None:
        widths = [487 / count] * count
        if count == 2:
            widths = [180, 307]
    item = Table([[Paragraph(markup(str(cell)), SMALL) for cell in row] for row in rows], colWidths=widths, hAlign="LEFT")
    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d8eeea")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), .7, TEAL),
    ]))
    return [item, Spacer(1, 14)]


def field(c, page, key, label, y):
    c.setFont("Helvetica-Bold", 10); c.setFillColor(INK)
    c.drawString(56, y + 8, label)
    c.acroForm.textfield(name=f"{page}_{key}", tooltip=label, x=194, y=y, width=346, height=26,
                        value=DETAILS[key], fontName="Helvetica", fontSize=10,
                        borderWidth=.5, borderColor=colors.HexColor("#a4b9c2"),
                        fillColor=colors.HexColor("#f4f8fa"), textColor=INK, forceBorder=True)


def front_page(c, kind):
    if kind == "cover":
        c.setFillColor(TEAL); c.rect(0, 785, 595, 57, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 13)
        c.drawString(54, 807, "MACHINE LEARNING | COLLEGE PROJECT REPORT")
        p = Paragraph(TITLE, ParagraphStyle("cover", parent=HEAD, fontSize=27, leading=34))
        _, h = p.wrap(487, 200); p.drawOn(c, 54, 724-h)
        c.setFont("Helvetica", 12); c.setFillColor(TEAL)
        c.drawString(54, 495, "Final project submission | Editable academic details below")
        start = 424
    elif kind == "certificate":
        p = Paragraph("Certificate page", HEAD); p.wrap(487, 100); p.drawOn(c, 54, 748)
        copy = "<b>Unsigned certificate template - institutional approval required.</b><br/><br/>The following project is submitted for academic review. Certification of completion, supervision and acceptance must be provided by the authorized college representatives after their review. No approval, guide identity or signature is asserted by this generated template."
        p = Paragraph(copy, BODY); _, h = p.wrap(487, 170); p.drawOn(c, 54, 711-h)
        start = 505
    else:
        p = Paragraph("Acknowledgement", HEAD); p.wrap(487, 100); p.drawOn(c, 54, 748)
        copy = "<b>Editable student acknowledgement.</b><br/><br/>I acknowledge the academic learning environment provided by my institution and the publicly available documentation and open-source software used to build this project. I also recognize the importance of careful testing, reproducible data and honest reporting of limitations.<br/><br/>A guide has not been assigned in the supplied details. A personal acknowledgement of supervision or assistance may be added by the student when those details are confirmed."
        p = Paragraph(copy, BODY); _, h = p.wrap(487, 230); p.drawOn(c, 54, 711-h)
        start = 450
    labels = {"name":"Student name", "roll_number":"Register / roll number", "college":"College", "degree":"Department / degree", "guide":"Guide", "academic_year":"Academic year"}
    for i, (key, label) in enumerate(labels.items()):
        field(c, kind, key, label, start - i * 42)
    c.setFont("Helvetica", 9); c.setFillColor(INK)
    if kind == "certificate":
        c.drawString(54, 160, "Guide / authorized reviewer signature: __________________________")
        c.drawString(54, 125, "Head of department signature: _________________________________")
        c.drawString(54, 90, "Date and institutional seal: ____________________________________")
    else:
        c.drawString(54, 115, "Identity fields are editable PDF form fields. Source files are included.")
        c.drawString(54, 92, "Synthetic demonstration data; no real customer outcomes are claimed.")


def page_blocks(text):
    blocks = []
    # Headings and figure directives are blocks even without blank Markdown lines.
    text = re.sub(r"(?m)^(#{1,2} .+|@\w+(?: .+)?|!\[.*?\]\(.*?\))$", r"\n\n\1\n\n", text)
    for part in re.split(r"\n\s*\n", text.strip()):
        if part.startswith("# "):
            blocks.append(Paragraph(markup(part[2:]), HEAD))
        elif part.startswith("## "):
            blocks.append(Paragraph(markup(part[3:]), SUB))
        elif part.startswith("@diagram "):
            blocks.extend([Diagram(part.split()[1]), Spacer(1, 10)])
        elif part.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", part)
            path = (DOC / match[2]).resolve()
            w, h = ImageReader(str(path)).getSize()
            width = min(487, w); height = width * h / w
            if height > 325:
                width *= 325 / height; height = 325
            blocks.extend([Image(str(path), width=width, height=height, hAlign="CENTER"), Spacer(1, 8), Paragraph(markup(match[1]) + ". Source: actual project output.", SMALL), Spacer(1, 10)])
        elif part.startswith("@comparison"):
            import csv
            rows = list(csv.DictReader((BASE / "outputs/model_comparison.csv").open()))
            blocks.extend(table([["Model", "Groups", "Coverage", "Silhouette", "DBI", "CH"]] + [[r["Model"], r["Clusters"], f"{float(r['Coverage']):.1%}", f"{float(r['Silhouette']):.4f}", f"{float(r['Davies-Bouldin']):.4f}", f"{float(r['Calinski-Harabasz']):.2f}"] for r in rows], [117,48,67,85,75,95]))
        elif part.startswith("@profiles"):
            import csv
            rows = list(csv.DictReader((BASE / "outputs/cluster_centers.csv").open()))
            blocks.extend(table([["Group", "Customers", "Income (k$)", "Spending", "Recency"]] + [[r["Cluster"],r["Customers"],f"{float(r['Annual Income']):.2f}",f"{float(r['Spending Score']):.2f}",f"{float(r['Recency']):.2f}"] for r in rows]))
        elif part.startswith("|"):
            blocks.extend(table([[cell.strip() for cell in line.strip().strip("|").split("|")] for line in part.splitlines()]))
        else:
            blocks.append(Paragraph(markup(part.replace("\n", " ")), BODY))
    return blocks


def main():
    pages = (DOC / "Project_Report_Source.md").read_text(encoding="utf-8").split("---PAGE---")
    assert len(pages) == 44, len(pages)
    path = DOC / "AI_Customer_Segmentation_Project_Report.pdf"
    c = canvas.Canvas(str(path), pagesize=(595, 842))
    c.setTitle(TITLE); c.setAuthor(DETAILS["name"])
    for number, text in enumerate(pages, 1):
        c.setFillColor(INK); c.setFont("Helvetica", 8)
        if number > 1:
            c.drawString(54, 809, "AI CUSTOMER SEGMENTATION | PROJECT REPORT")
            c.setStrokeColor(TEAL); c.line(54, 797, 541, 797)
        c.drawString(54, 32, "CUSTOMER ANALYTICS | COLLEGE SUBMISSION")
        c.drawRightString(541, 32, f"{number} / {len(pages)}")
        if number <= 3:
            front_page(c, ["cover", "certificate", "acknowledgement"][number - 1])
        else:
            blocks = page_blocks(text)
            frame = Frame(54, 57, 487, 719, leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
            frame.addFromList(blocks, c)
            if blocks:
                raise RuntimeError(f"Page {number} overflow: {len(blocks)} remaining blocks")
        c.showPage()
    c.save()
    reader = PdfReader(path)
    assert len(reader.pages) == 44
    fields = reader.get_fields()
    assert len(fields) == 18
    for prefix in ["cover", "certificate", "acknowledgement"]:
        for key, value in DETAILS.items():
            assert fields[f"{prefix}_{key}"]["/V"] == value
    shutil.copy2(path, DOC / "Project_Report.pdf")
    shutil.copy2(path, BASE / path.name)
    print(f"Report: {len(reader.pages)} pages; {len(fields)} verified editable fields; {path}")


if __name__ == "__main__":
    main()
