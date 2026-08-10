"""Renderer PDF officiel du contrat particulier — contrat formel simple (3 pages)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from services.platform_billing.partner_agreement_particular_content import (
    ParticularAgreementContent,
)

INK = colors.HexColor("#111111")
MUTED = colors.HexColor("#444444")
PAGE_W, PAGE_H = A4
MARGIN_X = 20 * mm
MARGIN_TOP = 16 * mm
MARGIN_BOTTOM = 14 * mm
FOOTER_H = 9 * mm
_LOGO_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "assets" / "lirie" / "logo-lirie.png",
    Path("/app/assets/lirie/logo-lirie.png"),
)


class PartnerAgreementLayoutError(Exception):
    """Le PDF particulier ne respecte pas la contrainte de pagination."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=INK,
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=5,
            leading=16,
        ),
        "sub": ParagraphStyle(
            "PSub",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=5,
            leading=11,
        ),
        "meta": ParagraphStyle(
            "PMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=3,
            leading=11,
        ),
        "article": ParagraphStyle(
            "PArticle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=3,
            leading=13,
        ),
        "body": ParagraphStyle(
            "PBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=3,
            leading=11.5,
        ),
        "partyHead": ParagraphStyle(
            "PPartyHead",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=INK,
            spaceBefore=3,
            spaceAfter=2,
            leading=12,
        ),
        "clauseTitle": ParagraphStyle(
            "PClauseTitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=INK,
            spaceBefore=5,
            spaceAfter=1.5,
            leading=12,
        ),
        "intro": ParagraphStyle(
            "PIntro",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5,
            leading=11.5,
        ),
        "term": ParagraphStyle(
            "PTerm",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2,
            leading=11.5,
        ),
        "principle": ParagraphStyle(
            "PPrinciple",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2,
            leading=11.5,
            leftIndent=12,
            firstLineIndent=-12,
        ),
        "sigHead": ParagraphStyle(
            "PSigHead",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=INK,
            spaceBefore=0,
            spaceAfter=4,
            leading=12,
        ),
        "sigBody": ParagraphStyle(
            "PSigBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=INK,
            leading=12,
            spaceAfter=2,
        ),
        "sigAttest": ParagraphStyle(
            "PSigAttest",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=MUTED,
            leading=10,
            spaceAfter=4,
        ),
        "sigLine": ParagraphStyle(
            "PSigLine",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=INK,
            leading=12,
            spaceBefore=2,
            spaceAfter=2,
        ),
    }


def _esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_esc(text), style)


def _logo() -> Image | None:
    """Logo centré, proportions d'origine conservées."""
    for path in _LOGO_CANDIDATES:
        if not path.is_file():
            continue
        # Largeur cible ; la hauteur suit le ratio natif (ex. ~2.31:1).
        target_w = 38 * mm
        img = Image(str(path))
        native_w = float(img.imageWidth or 1)
        native_h = float(img.imageHeight or 1)
        ratio = native_h / native_w if native_w else 1.0
        img.drawWidth = target_w
        img.drawHeight = target_w * ratio
        img.hAlign = "CENTER"
        return img
    return None


def _rule() -> HRFlowable:
    return HRFlowable(
        width="100%",
        thickness=0.5,
        color=colors.HexColor("#888888"),
        spaceBefore=4,
        spaceAfter=7,
    )


def _signature_column(sig, styles: dict) -> list:
    parts: list = [
        _p(sig.side, styles["sigHead"]),
        _p(sig.name, styles["sigBody"]),
        _p(sig.title, styles["sigBody"]),
    ]
    if sig.power_attestation:
        parts.append(_p(sig.power_attestation, styles["sigAttest"]))
    else:
        parts.append(Spacer(1, 6 * mm))
    parts.extend(
        [
            Spacer(1, 3 * mm),
            _p("Lieu et date :", styles["sigLine"]),
            _p("________________________________", styles["sigLine"]),
            Spacer(1, 5 * mm),
            _p("Signature :", styles["sigLine"]),
            Spacer(1, 12 * mm),
            _p("________________________________", styles["sigLine"]),
        ]
    )
    if sig.co_signatory_name:
        co = sig.co_signatory_name
        if sig.co_signatory_title:
            co = f"{co}, {sig.co_signatory_title}"
        parts.extend(
            [
                Spacer(1, 4 * mm),
                _p(f"Co-signataire : {co}", styles["sigBody"]),
                Spacer(1, 3 * mm),
                _p("Signature :", styles["sigLine"]),
                Spacer(1, 10 * mm),
                _p("________________________________", styles["sigLine"]),
            ]
        )
    return parts


def _signatures_block(content: ParticularAgreementContent, styles: dict) -> Table:
    """Deux colonnes côte à côte, sans cadre ni filet."""
    left = _signature_column(content.signatures[0], styles)
    right = _signature_column(content.signatures[1], styles)
    gap = 8 * mm
    col_w = (PAGE_W - 2 * MARGIN_X - gap) / 2
    tbl = Table([[left, "", right]], colWidths=[col_w, gap, col_w])
    tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return tbl


def _build_story(content: ParticularAgreementContent) -> list:
    styles = _styles()
    story: list = []

    logo = _logo()
    if logo:
        story.append(Spacer(1, 1 * mm))
        story.append(logo)
        story.append(Spacer(1, 5 * mm))

    story.append(_p(content.title, styles["title"]))
    story.append(_p(content.subtitle, styles["sub"]))
    story.append(_p(f"Référence : {content.reference}", styles["meta"]))
    story.append(
        _p(
            f"Date d'effet commerciale : {content.effective_date_fr}",
            styles["meta"],
        )
    )
    story.append(_p(f"Version : {content.particular_version}", styles["meta"]))
    story.append(Spacer(1, 2 * mm))
    story.append(_rule())
    story.append(_p(content.pack_note, styles["intro"]))

    story.append(_p("Article 1 — Identification des Parties", styles["article"]))
    for col in content.parties:
        story.append(_p(col.header, styles["partyHead"]))
        for line in col.lines:
            story.append(_p(line, styles["body"]))

    story.append(_p("Article 2 — Conditions particulières", styles["article"]))
    for row in content.commercial_terms:
        story.append(
            Paragraph(
                f"<b>{_esc(row.label)}</b> : {_esc(row.value)}",
                styles["term"],
            )
        )
    if content.special_conditions:
        story.append(Spacer(1, 2 * mm))
        story.append(
            _p("Conditions particulières complémentaires", styles["clauseTitle"])
        )
        for line in content.special_conditions[:6]:
            story.append(_p(f"— {line}", styles["body"]))

    story.append(_p(content.key_principles_title, styles["article"]))
    for i, principle in enumerate(content.key_principles, start=1):
        story.append(_p(f"{i}.  {principle}", styles["principle"]))

    # ——— Page 2 ———
    story.append(PageBreak())
    story.append(_p("Article 4 — Clauses générales essentielles", styles["article"]))
    story.append(_p(content.clauses_intro, styles["intro"]))
    for clause in content.clauses:
        story.append(
            KeepTogether(
                [
                    _p(clause.title, styles["clauseTitle"]),
                    _p(clause.body, styles["body"]),
                ]
            )
        )

    # ——— Page 3 ———
    story.append(PageBreak())
    story.append(_p("Article 5 — Protection des données", styles["article"]))
    for role in content.data_protection_roles:
        story.append(
            Paragraph(
                f"<b>{_esc(role.treatment)}</b> : {_esc(role.role)}",
                styles["term"],
            )
        )
    story.append(Spacer(1, 2 * mm))
    story.append(_p(content.data_protection_summary, styles["body"]))
    story.append(_p(content.gps_summary, styles["body"]))
    story.append(_p(content.providers_summary, styles["body"]))

    story.append(_p("Article 6 — Documents contractuels incorporés", styles["article"]))
    story.append(
        _p(
            "Le Partenaire reconnaît avoir reçu et accepté, avant la signature :",
            styles["body"],
        )
    )
    for doc in content.incorporated_documents:
        story.append(_p(f"• {doc.label}", styles["body"]))
        story.append(_p(f"  Version : {doc.version}", styles["body"]))
    story.append(Spacer(1, 1.5 * mm))
    story.append(_p(content.acceptance_clause, styles["body"]))

    story.append(_p("Article 7 — Signatures", styles["article"]))
    story.append(_p(content.signature_intro, styles["intro"]))
    story.append(Spacer(1, 2 * mm))
    story.append(_signatures_block(content, styles))

    return story


def count_pdf_pages(pdf_bytes: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(BytesIO(pdf_bytes)).pages)


def build_particular_pdf_bytes(content: ParticularAgreementContent) -> bytes:
    buffer = BytesIO()

    def _on_page(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#AAAAAA"))
        canvas.setLineWidth(0.4)
        y_line = MARGIN_BOTTOM - 1 * mm
        canvas.line(MARGIN_X, y_line, PAGE_W - MARGIN_X, y_line)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            MARGIN_X,
            MARGIN_BOTTOM - 5 * mm,
            f"LIRIE · {content.reference}",
        )
        canvas.drawRightString(
            PAGE_W - MARGIN_X,
            MARGIN_BOTTOM - 5 * mm,
            f"Page {doc.page} sur 3",
        )
        canvas.restoreState()

    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        title=content.title,
        author="LIRIE",
        creator="LIRIE-partner-particular",
    )
    frame = Frame(
        MARGIN_X,
        MARGIN_BOTTOM + FOOTER_H - 2 * mm,
        PAGE_W - 2 * MARGIN_X,
        PAGE_H - MARGIN_TOP - MARGIN_BOTTOM - FOOTER_H + 2 * mm,
        id="normal",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_on_page)])
    doc.build(_build_story(content))
    pdf_bytes = buffer.getvalue()
    pages = count_pdf_pages(pdf_bytes)
    if pages != 3:
        raise PartnerAgreementLayoutError(
            f"Le contrat particulier doit contenir exactement 3 pages, obtenu : {pages}"
        )
    from pypdf import PdfReader

    text = "\n".join(
        (p.extract_text() or "") for p in PdfReader(BytesIO(pdf_bytes)).pages
    )
    if "BROUILLON" in text.upper():
        raise PartnerAgreementLayoutError(
            "Le PDF officiel ne doit pas contenir le mot « BROUILLON »."
        )
    return pdf_bytes
