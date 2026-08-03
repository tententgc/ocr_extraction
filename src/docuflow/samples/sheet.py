"""Thin top-down drawing helper over a ReportLab canvas.

ReportLab measures from the bottom-left corner, which makes document layouts
awkward to write. `Sheet` flips to a top-down cursor and adds the few primitives
the sample layouts need.
"""

from __future__ import annotations

from dataclasses import dataclass

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from docuflow.samples.fonts import register_thai_font

BLACK = HexColor("#111111")
GREY = HexColor("#666666")
LIGHT = HexColor("#DDDDDD")


@dataclass
class Sheet:
    """A page being drawn, with a top-down `y` cursor."""

    canvas: Canvas
    width: float
    height: float
    margin: float = 48.0
    y: float = 0.0
    base_font: str = "Helvetica"
    bold_font: str = "Helvetica-Bold"

    @classmethod
    def create(
        cls,
        path: str,
        page_size: tuple[float, float] = A4,
        margin: float = 48.0,
        thai: bool = False,
    ) -> Sheet:
        canvas = Canvas(path, pagesize=page_size)
        width, height = page_size
        base, bold = "Helvetica", "Helvetica-Bold"
        if thai:
            thai_font = register_thai_font()
            if thai_font:
                # The Thai fonts available on macOS ship a single weight, so bold
                # falls back to the same face rather than a synthetic one.
                base = bold = thai_font
        sheet = cls(canvas=canvas, width=width, height=height, margin=margin, base_font=base, bold_font=bold)
        sheet.y = height - margin
        return sheet

    # --- geometry ----------------------------------------------------------
    @property
    def left(self) -> float:
        return self.margin

    @property
    def right(self) -> float:
        return self.width - self.margin

    @property
    def content_width(self) -> float:
        return self.width - 2 * self.margin

    def down(self, amount: float) -> None:
        self.y -= amount

    def needs_page(self, required: float = 90.0) -> bool:
        return self.y < self.margin + required

    def new_page(self) -> None:
        self.canvas.showPage()
        self.y = self.height - self.margin

    # --- text --------------------------------------------------------------
    def text(
        self,
        value: str,
        x: float | None = None,
        size: float = 9.5,
        bold: bool = False,
        color: Color = BLACK,
        align: str = "left",
        leading: float | None = None,
    ) -> None:
        """Draw one line at the cursor and advance."""
        if value is None:
            value = ""
        self.canvas.setFont(self.bold_font if bold else self.base_font, size)
        self.canvas.setFillColor(color)
        position = self.left if x is None else x
        if align == "right":
            self.canvas.drawRightString(position, self.y, str(value))
        elif align == "center":
            self.canvas.drawCentredString(position, self.y, str(value))
        else:
            self.canvas.drawString(position, self.y, str(value))
        self.down(leading if leading is not None else size + 3.5)

    def at(
        self,
        value: str,
        x: float,
        y: float,
        size: float = 9.5,
        bold: bool = False,
        color: Color = BLACK,
        align: str = "left",
    ) -> None:
        """Draw at an absolute position without moving the cursor."""
        self.canvas.setFont(self.bold_font if bold else self.base_font, size)
        self.canvas.setFillColor(color)
        if align == "right":
            self.canvas.drawRightString(x, y, str(value))
        elif align == "center":
            self.canvas.drawCentredString(x, y, str(value))
        else:
            self.canvas.drawString(x, y, str(value))

    def paragraph(self, value: str, size: float = 9.5, leading: float = 13.5, indent: float = 0.0) -> None:
        """Word-wrap a block of prose across pages."""
        self.canvas.setFont(self.base_font, size)
        max_width = self.content_width - indent
        words = str(value).split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if self.canvas.stringWidth(candidate, self.base_font, size) <= max_width:
                line = candidate
                continue
            if self.needs_page(leading * 2):
                self.new_page()
                self.canvas.setFont(self.base_font, size)
            self.at(line, self.left + indent, self.y, size=size)
            self.down(leading)
            line = word
        if line:
            if self.needs_page(leading * 2):
                self.new_page()
            self.at(line, self.left + indent, self.y, size=size)
            self.down(leading)

    # --- shapes ------------------------------------------------------------
    def rule(self, color: Color = LIGHT, thickness: float = 0.7, gap: float = 6.0) -> None:
        self.canvas.setStrokeColor(color)
        self.canvas.setLineWidth(thickness)
        self.canvas.line(self.left, self.y, self.right, self.y)
        self.down(gap)

    def band(self, height: float, color: Color, x: float | None = None, width: float | None = None) -> None:
        """Filled rectangle whose top edge is the cursor; advances past it."""
        self.canvas.setFillColor(color)
        left = self.left if x is None else x
        box_width = self.content_width if width is None else width
        self.canvas.rect(left, self.y - height, box_width, height, stroke=0, fill=1)
        self.canvas.setFillColor(BLACK)

    def box(self, x: float, y: float, width: float, height: float, stroke: Color = LIGHT, fill: Color | None = None) -> None:
        self.canvas.setStrokeColor(stroke)
        self.canvas.setLineWidth(0.7)
        if fill is not None:
            self.canvas.setFillColor(fill)
        self.canvas.rect(x, y, width, height, stroke=1, fill=1 if fill is not None else 0)
        self.canvas.setFillColor(BLACK)

    def checkbox(self, x: float, y: float, checked: bool, size: float = 8.0) -> None:
        self.box(x, y, size, size)
        if checked:
            self.canvas.setStrokeColor(BLACK)
            self.canvas.setLineWidth(1.1)
            self.canvas.line(x + 1.5, y + size / 2, x + size / 2.4, y + 1.8)
            self.canvas.line(x + size / 2.4, y + 1.8, x + size - 1.2, y + size - 1.5)

    def save(self) -> None:
        self.canvas.save()
