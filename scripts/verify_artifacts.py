"""Render and check the final PDF/PPTX and all local Markdown file links."""
import json
import re
import zipfile
import csv
from pathlib import Path
from xml.etree import ElementTree as ET
import pymupdf as fitz
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader

BASE = Path(__file__).resolve().parents[1]
DOC = BASE / "documentation"
QA = DOC / ".qa"
QA.mkdir(exist_ok=True)
pdf = DOC / "AI_Customer_Segmentation_Project_Report.pdf"
reader = PdfReader(pdf)
assert len(reader.pages) == 44
assert len(reader.get_fields()) == 18
render_dir = QA / "report"
render_dir.mkdir(exist_ok=True)
document = fitz.open(pdf)
word_counts = []
for i, page in enumerate(document):
    word_counts.append(len(page.get_text().split()))
    assert word_counts[-1] > 35, (i+1, "Unexpected empty page")
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                x0,y0,x1,y1=span["bbox"]
                assert x0 >= -1 and y0 >= -1 and x1 <= page.rect.width+1 and y1 <= page.rect.height+1, (i+1, span["text"])
    page.get_pixmap(matrix=fitz.Matrix(1.2,1.2), annots=True).save(render_dir / f"page-{i+1:02}.png")

# Check content structure as well as page count to catch Markdown rendering errors.
assert sum(len(page.get_images()) for page in document) >= 8
assert "@diagram" not in " ".join(page.get_text() for page in document)

def contact(paths, output, cols, width=320):
    tiles=[]
    for path in paths:
        image=Image.open(path).convert("RGB")
        image.thumbnail((width,460))
        tile=Image.new("RGB",(width+20,490),"#dce5eb")
        tile.paste(image,((width+20-image.width)//2,25))
        ImageDraw.Draw(tile).text((10,5),path.stem,fill="black")
        tiles.append(tile)
    sheet=Image.new("RGB",((width+20)*cols,490*((len(tiles)+cols-1)//cols)),"white")
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%cols)*(width+20),(i//cols)*490))
    sheet.save(output)

paths=sorted(render_dir.glob("page-*.png"))
for start in range(0,len(paths),8):
    contact(paths[start:start+8],QA/f"report-contact-{start//8+1}.png",4)

pptx=DOC/"AI_Customer_Segmentation_Presentation.pptx"
with zipfile.ZipFile(pptx) as archive:
    assert archive.testzip() is None
    slides=[name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml",name)]
    assert len(slides)==15
    for name in slides:
        root=ET.fromstring(archive.read(name))
        text=" ".join(root.itertext())
        assert len(text)>20
    tables=sum(archive.read(name).count(b'<a:tbl>') for name in slides)
    assert tables>=4
    comparison=ET.fromstring(archive.read("ppt/slides/slide11.xml"))
    ns={"a":"http://schemas.openxmlformats.org/drawingml/2006/main"}
    rows=comparison.findall(".//a:tbl/a:tr",ns)
    assert len(rows)==5
    assert ["".join(cell.itertext()) for cell in rows[0].findall("a:tc",ns)] == ["Model","Groups","Coverage","Silhouette","DBI","CH"]

broken=[]
for source in [BASE/"README.md",BASE/"VIVA_GUIDE_FINAL.md",*DOC.glob("*.md")]:
    if not source.exists():continue
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',source.read_text(encoding="utf-8")):
        if target.startswith(("http", "#")):continue
        path=(source.parent/target.split('#')[0]).resolve()
        if not path.exists():broken.append(str(path))
viva = (BASE/"VIVA_GUIDE_FINAL.md").read_text(encoding="utf-8")
assert [int(value) for value in re.findall(r"(?m)^### (\d+)\.", viva)] == list(range(1, 51))
assert (DOC/"Viva_Guide.md").read_text(encoding="utf-8") == viva
for original, aliases in [(pdf, [BASE/pdf.name, DOC/"Project_Report.pdf"]),
                          (pptx, [BASE/pptx.name, DOC/"Presentation.pptx"])]:
    for alias in aliases:
        assert original.read_bytes() == alias.read_bytes(), f"Stale artifact copy: {alias}"
screenshots=list((DOC/"screenshots").glob("*.png"))
assert len(screenshots)==8
for path in screenshots:
    with Image.open(path) as screenshot:
        screenshot.verify()
    assert path.read_bytes() == (BASE/"screenshots"/path.name).read_bytes()
for filename, expected in [("Mall_Customers.csv",2000),("Transactions.csv",32800)]:
    with (BASE/"dataset"/filename).open(encoding="utf-8-sig",newline="") as handle:
        assert sum(1 for row in csv.DictReader(handle)) == expected
result={"pdf_pages":44,"editable_pdf_fields":18,"report_word_count":sum(word_counts),"slide_count":15,
        "native_tables":tables,"rendered_report_pages":len(paths),"viva_questions":50,
        "screenshots":8,"customers":2000,"transactions":32800,"artifact_copies_match":True,"broken_links":broken}
(QA/"artifact_checks.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
if broken:raise RuntimeError("Broken documentation links")
