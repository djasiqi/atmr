"""PDF officiel des conditions PORTAL (CGU / CGV) — logo LIRIE + corps canonique."""

from __future__ import annotations

from html import escape as html_escape
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from models.client_terms_acceptance import (
    DOCUMENT_TERMS_OF_SERVICE,
    DOCUMENT_TRANSPORT_TERMS,
)
from services.legal.portal_terms_catalog import PublishedTerms

_LOGO_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "assets" / "lirie" / "logo-lirie.png",
    Path("/app/assets/lirie/logo-lirie.png"),
    Path("/app/backend/assets/lirie/logo-lirie.png"),
)

# Cadre d’affichage max — le ratio natif du PNG (≈ 2.31) est toujours respecté.
_LOGO_MAX_WIDTH = 3.6 * cm
_LOGO_MAX_HEIGHT = 1.55 * cm

_INK = colors.HexColor("#0f172a")
_MUTED = colors.HexColor("#475569")
_ACCENT = colors.HexColor("#0b5cab")
_RULE = colors.HexColor("#cbd5e1")

_DOCUMENT_TITLES = {
    DOCUMENT_TERMS_OF_SERVICE: "Conditions générales d’utilisation du compte client privé",
    DOCUMENT_TRANSPORT_TERMS: (
        "Conditions générales de réservation et de transport — client privé"
    ),
}


def resolve_lirie_logo_path() -> Path | None:
    for candidate in _LOGO_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _logo_flowable(
    *,
    max_width: float = _LOGO_MAX_WIDTH,
    max_height: float = _LOGO_MAX_HEIGHT,
) -> Image | None:
    """Image LIRIE sans déformation (ratio natif du fichier officiel)."""
    logo_path = resolve_lirie_logo_path()
    if logo_path is None:
        return None
    try:
        native_w, native_h = ImageReader(str(logo_path)).getSize()
        if not native_w or not native_h:
            return None
        aspect = float(native_h) / float(native_w)
        width = float(max_width)
        height = width * aspect
        if height > max_height:
            height = float(max_height)
            width = height / aspect
        logo = Image(str(logo_path), width=width, height=height)
        logo.hAlign = "LEFT"
        return logo
    except Exception:
        return None


def portal_terms_pdf_filename(spec: PublishedTerms) -> str:
    kind = {
        DOCUMENT_TERMS_OF_SERVICE: "CGU_compte_client_prive",
        DOCUMENT_TRANSPORT_TERMS: "CGV_reservation_transport",
    }.get(spec.document_type, spec.document_type)
    return f"LIRIE_{kind}_v{spec.terms_version}.pdf"


def portal_terms_pdf_title(spec: PublishedTerms) -> str:
    return _DOCUMENT_TITLES.get(
        spec.document_type,
        f"Document contractuel LIRIE ({spec.document_type})",
    )


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PortalTermsTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=_INK,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "PortalTermsMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=_MUTED,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "PortalTermsBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=_INK,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        "heading": ParagraphStyle(
            "PortalTermsHeading",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=_INK,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "PortalTermsFooter",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            textColor=_MUTED,
            alignment=TA_CENTER,
        ),
    }


def _paragraph_chunks(canonical_body: str) -> list[tuple[str, str]]:
    """Découpe le corps en (kind, text) avec kind in {heading, body}."""
    chunks: list[tuple[str, str]] = []
    for raw in (canonical_body or "").replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        # Titres numérotés « 1. … » / « 12. … »
        if len(stripped) < 120 and stripped[:1].isdigit() and ". " in stripped[:6]:
            chunks.append(("heading", stripped))
        else:
            chunks.append(("body", stripped))
    return chunks


def build_portal_terms_pdf_bytes(spec: PublishedTerms) -> bytes:
    """Génère un PDF A4 officiel à partir d’une version canonique figée."""
    buffer = BytesIO()
    styles = _styles()
    title = portal_terms_pdf_title(spec)
    hash_short = (spec.terms_hash or "")[:16]

    def _footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(_RULE)
        canvas.setLineWidth(0.4)
        y = 1.35 * cm
        canvas.line(1.8 * cm, y + 0.55 * cm, A4[0] - 1.8 * cm, y + 0.55 * cm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(_MUTED)
        canvas.drawCentredString(
            A4[0] / 2,
            y,
            (
                f"LIRIE — document officiel · {spec.document_type} "
                f"v{spec.terms_version} · empreinte {hash_short}… · "
                f"www.lirie.ch · page {doc.page}"
            ),
        )
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=2.2 * cm,
        title=title,
        author="LIRIE",
        subject=f"{spec.document_type} {spec.terms_version}",
    )

    story: list = []
    logo = _logo_flowable()
    if logo is not None:
        story.append(logo)
        story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph(html_escape(title), styles["title"]))
    story.append(
        Paragraph(
            html_escape(
                f"Version {spec.terms_version} · locale {spec.locale} · "
                f"type {spec.document_type}"
            ),
            styles["meta"],
        )
    )
    story.append(
        Paragraph(
            html_escape(f"Empreinte SHA-256 : {spec.terms_hash}"),
            styles["meta"],
        )
    )
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        HRFlowable(
            width="100%",
            thickness=0.8,
            color=_ACCENT,
            spaceBefore=2,
            spaceAfter=10,
        )
    )

    for kind, text in _paragraph_chunks(spec.canonical_body):
        safe = html_escape(text).replace("\n", "<br/>")
        if kind == "heading":
            story.append(Paragraph(safe, styles["heading"]))
        else:
            story.append(Paragraph(safe, styles["body"]))

    story.append(Spacer(1, 0.6 * cm))
    story.append(
        Paragraph(
            html_escape(
                "Document généré par la plateforme LIRIE à partir du texte "
                "canonique figé. Conservez ce fichier pour vos archives."
            ),
            styles["meta"],
        )
    )

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
