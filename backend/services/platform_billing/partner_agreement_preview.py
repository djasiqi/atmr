"""Prévisualisation brouillon : filigrane dynamique sur le PDF officiel stocké."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas


WATERMARK_TEXT = "BROUILLON — NE PAS SIGNER"


def _watermark_page_bytes(width: float, height: float) -> bytes:
    packet = BytesIO()
    c = canvas.Canvas(packet, pagesize=(width, height))
    c.saveState()
    c.setFillColor(Color(0.75, 0.15, 0.15, alpha=0.28))
    c.setFont("Helvetica-Bold", 28)
    c.translate(width / 2, height / 2)
    c.rotate(45)
    c.drawCentredString(0, 0, WATERMARK_TEXT)
    c.restoreState()
    c.save()
    packet.seek(0)
    return packet.read()


def apply_draft_watermark(pdf_bytes: bytes) -> bytes:
    """Retourne un PDF temporaire filigrané (non stocké, non hashé)."""
    reader = PdfReader(BytesIO(pdf_bytes))
    writer = PdfWriter()
    for page in reader.pages:
        box = page.mediabox
        width = float(box.width)
        height = float(box.height)
        wm_reader = PdfReader(BytesIO(_watermark_page_bytes(width, height)))
        page.merge_page(wm_reader.pages[0])
        writer.add_page(page)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()
