"""Reusable monochrome research-report layout, independent of numerical estimation."""

from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class ResearchReport:
    def __init__(self, title, subtitle, report_date, figures):
        self.title, self.figures = title, figures
        self.width, self.height = A4
        self.usable = self.width - 124
        self.styles = {
            "body": ParagraphStyle(
                "body",
                fontName="Times-Roman",
                fontSize=10.5,
                leading=14.1,
                alignment=TA_JUSTIFY,
                spaceAfter=8,
            ),
            "title": ParagraphStyle(
                "title",
                fontName="Times-Roman",
                fontSize=19,
                leading=23,
                alignment=TA_CENTER,
                spaceAfter=8,
            ),
            "subtitle": ParagraphStyle(
                "subtitle",
                fontName="Times-Roman",
                fontSize=11,
                leading=15,
                alignment=TA_CENTER,
                spaceAfter=5,
            ),
            "date": ParagraphStyle(
                "date",
                fontName="Times-Roman",
                fontSize=9,
                leading=12,
                alignment=TA_CENTER,
                spaceAfter=18,
                textColor=colors.HexColor("#444444"),
            ),
            "h1": ParagraphStyle(
                "h1",
                fontName="Times-Bold",
                fontSize=12.5,
                leading=16,
                spaceBefore=10,
                spaceAfter=8,
                keepWithNext=True,
            ),
            "h2": ParagraphStyle(
                "h2",
                fontName="Times-Bold",
                fontSize=10.5,
                leading=14,
                spaceBefore=8,
                spaceAfter=5,
                keepWithNext=True,
            ),
            "caption": ParagraphStyle(
                "caption", fontName="Times-Roman", fontSize=9, leading=11.5, spaceAfter=8
            ),
            "tablecap": ParagraphStyle(
                "tablecap",
                fontName="Times-Roman",
                fontSize=9,
                leading=11.5,
                spaceBefore=4,
                spaceAfter=5,
                keepWithNext=True,
            ),
            "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.8, leading=11.1),
            "head": ParagraphStyle("head", fontName="Times-Bold", fontSize=8.8, leading=11.1),
            "smallcell": ParagraphStyle(
                "smallcell", fontName="Times-Roman", fontSize=8.2, leading=10.2
            ),
            "smallhead": ParagraphStyle(
                "smallhead", fontName="Times-Bold", fontSize=8.2, leading=10.2
            ),
            "ref": ParagraphStyle(
                "ref", fontName="Times-Roman", fontSize=9, leading=11.7, spaceAfter=6
            ),
        }
        self.story = [
            self.paragraph(title, "title"),
            self.paragraph(subtitle, "subtitle"),
            self.paragraph(report_date.strftime("%d %B %Y"), "date"),
        ]

    def paragraph(self, text, style="body"):
        return Paragraph(escape(str(text)), self.styles[style])

    def p(self, text):
        self.story.append(self.paragraph(text))

    def h(self, text, level=1):
        self.story.append(self.paragraph(text, "h1" if level == 1 else "h2"))

    def page(self):
        self.story.append(PageBreak())

    def table(self, caption, headers, rows, proportions, compact=False):
        self.story.append(self.paragraph(caption, "tablecap"))
        head, cell = ("smallhead", "smallcell") if compact else ("head", "cell")
        data = [[self.paragraph(v, head) for v in headers]] + [
            [self.paragraph(v, cell) for v in row] for row in rows
        ]
        table = Table(
            data, colWidths=[self.usable * v / sum(proportions) for v in proportions], repeatRows=1
        )
        padding = 2.5 if compact else 5
        table.setStyle(
            TableStyle(
                [
                    ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), padding),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
                ]
            )
        )
        self.story.extend([table, Spacer(1, 7)])

    def figure(self, name, caption="", maximum_height=185):
        path = self.figures / name
        width, height = ImageReader(str(path)).getSize()
        scale = min(self.usable / width, maximum_height / height)
        image = Image(str(path), width=width * scale, height=height * scale)
        image.hAlign = "CENTER"
        elements = [Spacer(1, 3), image, Spacer(1, 4)]
        if caption:
            elements.append(self.paragraph(caption, "caption"))
        self.story.append(KeepTogether(elements))

    def reference(self, text):
        self.story.append(self.paragraph(text, "ref"))

    def _footer(self, canvas, document):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8.5)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(62, 30, self.title)
        canvas.drawRightString(self.width - 62, 30, str(document.page))
        canvas.restoreState()

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        document = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=62,
            rightMargin=62,
            topMargin=53,
            bottomMargin=50,
            title=self.title,
            author="Sebastian Jaramillo",
            subject="Bayesian lag-order comparison, model-averaged persistence, and finite-horizon shock responses",
        )
        document.build(self.story, onFirstPage=self._footer, onLaterPages=self._footer)
