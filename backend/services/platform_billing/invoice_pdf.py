"""PDF facture légale plateforme LIRIE → transporteur (coordonnées, lignes, QR-Bill)."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from decimal import Decimal
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    NextPageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from services.documents.pdf import (
    DEST_ADDR_MAX_WIDTH_MM,
    INVOICE_PAGE_LEFT_MARGIN_CM,
    INVOICE_PAGE_RIGHT_MARGIN_CM,
    INVOICE_PAGE_TOP_MARGIN_CM,
    _load_logo_ratio_safe,
    _make_invoice_doc_with_qrbill_page,
    _make_qr_bill_flowable,
    _svg_content_to_drawing,
)
from services.platform_billing.money import money_round_chf
from services.platform_billing.swiss_qr import (
    QrParty,
    SwissQrBillPayload,
    platform_qr_amount,
    render_swiss_qr_bill,
    resolve_platform_reference_mode,
)

logger = logging.getLogger(__name__)

_PDF_ROOT = Path("/app/uploads/platform_invoices")
_LOGO_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "assets" / "lirie" / "logo-lirie.png",
    Path("/app/assets/lirie/logo-lirie.png"),
    Path("/app/backend/assets/lirie/logo-lirie.png"),
)
_LIRIE_SITE_URL = "https://www.lirie.ch"
_MONTHS_FR = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)


def _qrr_check_digit(reference_base: str) -> int:
    """ISO 7064 MOD 10 récursif (27e chiffre QRR)."""
    accumulator = 10
    for digit_char in reference_base:
        if not digit_char.isdigit():
            raise ValueError(f"QRR doit être numérique: {reference_base}")
        digit = int(digit_char)
        accumulator = (accumulator + digit) % 10
        if accumulator == 0:
            accumulator = 10
    return (10 - accumulator) % 10


def build_platform_qrr_reference(
    *,
    invoice_number: str,
    issued_id: int,
    creditor_reference_base: str | None = "21",
) -> str:
    """Référence QRR 27 chiffres (base + numéro + id + check digit)."""
    base = re.sub(r"\D", "", creditor_reference_base or "21") or "21"
    inv_digits = re.sub(r"\D", "", invoice_number or "") or "0"
    id_digits = f"{int(issued_id):04d}"[-4:]
    # 26 chiffres avant check digit
    body = (base + inv_digits.zfill(20) + id_digits)[-26:].zfill(26)
    return body + str(_qrr_check_digit(body))


def build_platform_scor_reference(*, invoice_number: str) -> str:
    """Référence créancier SCOR (ISO 11649), ex. ``RF52 1EM2 0260 7006 1``."""
    from services.billing import generate_scor_reference

    return generate_scor_reference(invoice_number or "LIRIE")


def _fmt_chf(amount: Decimal | str | None) -> str:
    if amount is None or amount == "":
        return "0.00"
    return f"{money_round_chf(Decimal(str(amount))):.2f}"


def _fmt_tax_rate(rate: Decimal | str | None) -> str:
    """Affiche 8.1 plutôt que 8.1000."""
    if rate is None or rate == "":
        return "0"
    text = f"{Decimal(str(rate)):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _fmt_qty(qty: Decimal | str | int | None) -> str:
    """Quantité sans zéros inutiles : 1.0000 → 1, 1.50 → 1.5."""
    if qty is None or qty == "":
        return ""
    d = Decimal(str(qty))
    text = f"{d:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _resolve_lirie_logo_path() -> Path | None:
    for path in _LOGO_CANDIDATES:
        if path.is_file():
            return path
    return None


def _build_url_qr_png_bytes(url: str) -> bytes | None:
    """PNG brut d’un QR marketing (pour canvas pied de page)."""
    try:
        import qrcode

        qr = qrcode.QRCode(box_size=4, border=1)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        logger.exception("Échec génération QR marketing %s", url)
        return None


def _make_platform_promo_footer_callback(
    *,
    extra_on_page: Any | None = None,
) -> Any:
    """Dessine le QR LIRIE + lien en bas de page (zone fixe, hors flux)."""
    from reportlab.lib.utils import ImageReader

    qr_png = _build_url_qr_png_bytes(_LIRIE_SITE_URL)
    qr_size = 1.85 * cm
    baseline = 0.65 * cm

    def _draw(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        left = float(doc.leftMargin)
        page_w = float(doc.pagesize[0])
        right = page_w - float(doc.rightMargin)
        usable = right - left

        # Trait de séparation au-dessus du bloc pied
        y_rule = baseline + qr_size + 0.35 * cm
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(0.45)
        canvas.line(left, y_rule, left + usable, y_rule)

        text_x = left
        if qr_png:
            canvas.drawImage(
                ImageReader(BytesIO(qr_png)),
                left,
                baseline,
                width=qr_size,
                height=qr_size,
                mask="auto",
                preserveAspectRatio=True,
            )
            text_x = left + qr_size + 0.4 * cm

        canvas.setFillColor(colors.HexColor("#334155"))
        canvas.setFont("Helvetica", 7.5)
        line1 = (
            "LIRIE avec vous — scannez pour rejoindre la plateforme ou en savoir plus."
        )
        canvas.drawString(text_x, baseline + qr_size - 12, line1)
        canvas.setFillColor(colors.HexColor("#0f766e"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(text_x, baseline + 10, _LIRIE_SITE_URL)
        canvas.linkURL(
            _LIRIE_SITE_URL,
            (text_x, baseline + 4, text_x + 4.2 * cm, baseline + 20),
            relative=0,
        )
        canvas.restoreState()

        if extra_on_page is not None:
            extra_on_page(canvas, doc)

    return _draw


def _addr_block(
    *,
    legal_name: str,
    street: str,
    building: str | None,
    postal: str,
    city: str,
    country: str,
    uid: str | None = None,
    vat: str | None = None,
    email: str | None = None,
) -> str:
    street_line = " ".join(p for p in [street, building or ""] if p).strip()
    city_line = " ".join(p for p in [postal, city] if p).strip()
    parts = [xml_escape(legal_name or "")]
    if street_line:
        parts.append(xml_escape(street_line))
    if city_line:
        parts.append(xml_escape(city_line))
    if country:
        parts.append(xml_escape(country))
    if uid:
        parts.append(xml_escape(f"IDE : {uid}"))
    if vat:
        parts.append(xml_escape(f"N° TVA : {vat}"))
    if email:
        parts.append(xml_escape(email))
    return "<br/>".join(parts)


def generate_platform_invoice_pdf_bytes(
    *,
    invoice_number: str,
    issued_at: datetime | None,
    due_at: datetime | None,
    period_year: int,
    period_month: int,
    creditor_snap: dict[str, Any],
    debtor_snap: dict[str, Any],
    lines: list[dict[str, Any]],
    subtotal: Decimal,
    tax_rate: Decimal,
    tax_amount: Decimal,
    total: Decimal,
    qr_amount: Decimal,
    qr_reference: str | None,
    payment_reference_mode: str = "QRR",
    iban: str,
    payment_terms_days: int = 30,
    preview: bool = False,
) -> bytes:
    """Génère le PDF facture (page détail + page QR-Bill suisse).

    Si preview=True : filigrane, n° APERÇU, aucune page QR payable.
    """
    display_number = "APERÇU" if preview else invoice_number
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "PlatNormal",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=4,
        fontName="Helvetica",
        leading=11,
    )
    label_style = ParagraphStyle(
        "PlatLabel",
        parent=normal,
        fontSize=8,
        spaceAfter=2,
    )
    right_style = ParagraphStyle(
        "PlatRight",
        parent=normal,
        alignment=TA_RIGHT,
    )
    title_style = ParagraphStyle(
        "PlatTitle",
        parent=normal,
        fontSize=14,
        fontName="Helvetica-Bold",
        spaceAfter=8,
    )
    watermark_style = ParagraphStyle(
        "PlatWatermark",
        parent=normal,
        fontSize=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#b91c1c"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    buffer = BytesIO()

    def _draw_preview_banner(canvas, _doc):
        if not preview:
            return
        canvas.saveState()
        canvas.setFillColor(colors.Color(0.72, 0.11, 0.11, alpha=0.12))
        canvas.rect(0, A4[1] / 2 - 40, A4[0], 80, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#b91c1c"))
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawCentredString(A4[0] / 2, A4[1] / 2, "APERÇU — DOCUMENT NON PAYABLE")
        canvas.restoreState()

    # Réserve bas de page pour le QR marketing fixe (~3.2 cm)
    _PLATFORM_PROMO_BOTTOM_CM = 3.2
    on_content_page = _make_platform_promo_footer_callback(
        extra_on_page=_draw_preview_banner if preview else None,
    )

    doc = _make_invoice_doc_with_qrbill_page(
        buffer,
        top_margin_cm=INVOICE_PAGE_TOP_MARGIN_CM,
        bottom_margin_cm=_PLATFORM_PROMO_BOTTOM_CM,
        left_margin_cm=INVOICE_PAGE_LEFT_MARGIN_CM,
        right_margin_cm=INVOICE_PAGE_RIGHT_MARGIN_CM,
        on_first_page=on_content_page,
        on_later_pages=on_content_page,
        bottom_margin_later_cm=_PLATFORM_PROMO_BOTTOM_CM,
    )

    story: list[Any] = []

    creditor_html = _addr_block(
        legal_name=str(creditor_snap.get("legal_name") or "LIRIE"),
        street=str(creditor_snap.get("street_name") or ""),
        building=creditor_snap.get("building_number"),
        postal=str(creditor_snap.get("postal_code") or ""),
        city=str(creditor_snap.get("city") or ""),
        country=str(creditor_snap.get("country_code") or "CH"),
        uid=creditor_snap.get("uid_ide"),
        vat=creditor_snap.get("vat_number"),
    )
    debtor_html = _addr_block(
        legal_name=str(debtor_snap.get("legal_name") or "Entreprise"),
        street=str(debtor_snap.get("street_name") or ""),
        building=debtor_snap.get("building_number"),
        postal=str(debtor_snap.get("postal_code") or ""),
        city=str(debtor_snap.get("city") or ""),
        country=str(debtor_snap.get("country_code") or "CH"),
        uid=debtor_snap.get("uid_ide"),
        vat=debtor_snap.get("vat_number"),
        email=debtor_snap.get("billing_email"),
    )

    page_width_pt = A4[0]
    usable_width_pt = (
        page_width_pt
        - INVOICE_PAGE_LEFT_MARGIN_CM * cm
        - INVOICE_PAGE_RIGHT_MARGIN_CM * cm
    )
    dest_width_pt = DEST_ADDR_MAX_WIDTH_MM * mm
    company_width_pt = usable_width_pt - dest_width_pt

    # En-tête : logo seul, puis coordonnées LIRIE | facturé à
    logo_path = _resolve_lirie_logo_path()
    logo_img, _logo_w, _logo_h = _load_logo_ratio_safe(logo_path, 3.2 * cm)
    if logo_img is not None:
        logo_img.hAlign = "LEFT"
        story.append(logo_img)
        story.append(Spacer(1, 8))

    creditor_block = Table(
        [
            [Paragraph("<b>Émetteur :</b>", label_style)],
            [Paragraph(creditor_html, normal)],
        ],
        colWidths=[company_width_pt],
    )
    creditor_block.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    recipient_block = Table(
        [
            [Paragraph("<b>Facturé à :</b>", label_style)],
            [Paragraph(debtor_html, normal)],
        ],
        colWidths=[dest_width_pt],
    )
    recipient_block.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    header = Table(
        [[creditor_block, recipient_block]],
        colWidths=[company_width_pt, dest_width_pt],
    )
    header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, -1), 6),
                ("LEFTPADDING", (1, 0), (1, -1), 12 * mm),
                ("TOPPADDING", (1, 0), (1, -1), 0),
            ]
        )
    )
    story.append(header)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Facture plateforme LIRIE", title_style))
    if preview:
        story.append(Paragraph("APERÇU — DOCUMENT NON PAYABLE", watermark_style))

    if 1 <= period_month <= 12:
        period_label = f"{_MONTHS_FR[period_month - 1]} {period_year}"
    else:
        period_label = f"{period_month:02d}.{period_year}"

    issued_s = issued_at.strftime("%d.%m.%Y") if issued_at else "—"
    due_s = due_at.strftime("%d.%m.%Y") if due_at else "—"
    info_html = (
        f"<b>Numéro de facture :</b> {xml_escape(display_number)}<br/>"
        f"<b>Date d'émission :</b> {xml_escape(issued_s)}<br/>"
        f"<b>Date d'échéance :</b> {xml_escape(due_s)} "
        f"({int(payment_terms_days)} jours)<br/>"
        f"<b>Période de facturation :</b> {xml_escape(period_label)}"
    )
    if qr_reference and not preview:
        info_html += (
            f"<br/><b>Référence de paiement :</b> {xml_escape(str(qr_reference))}"
        )
    story.append(Paragraph(info_html, normal))
    story.append(Spacer(1, 14))

    table_data: list[list[Any]] = [
        [
            Paragraph("<b>Désignation</b>", normal),
            Paragraph("<b>Qté</b>", right_style),
            Paragraph("<b>Prix unit. (CHF)</b>", right_style),
            Paragraph("<b>Montant HT (CHF)</b>", right_style),
        ]
    ]
    if lines:
        from services.platform_billing.invoice_replace import sync_derived_line_label

        for ln in lines:
            line_type = str(ln.get("line_type") or "ADJUSTMENT")
            qty = ln.get("quantity")
            unit = ln.get("unit_amount")
            qty_dec = Decimal(str(qty)) if qty not in (None, "") else None
            unit_dec = Decimal(str(unit)) if unit not in (None, "") else None
            label = sync_derived_line_label(
                label=str(ln.get("label") or line_type or "Prestation"),
                line_type=line_type,
                quantity=qty_dec,
                unit_amount=unit_dec,
            )
            label = re.sub(
                r"\s*\(\s*created_at\s*\)", "", label, flags=re.IGNORECASE
            ).strip()
            amount = ln.get("amount")
            qty_s = _fmt_qty(qty)
            if (unit is None or unit == "") and qty not in (None, "", 0, "0"):
                try:
                    q = Decimal(str(qty))
                    if q > 0 and amount is not None:
                        unit = money_round_chf(Decimal(str(amount)) / q)
                except Exception:
                    unit = None
            unit_s = "" if unit is None or unit == "" else _fmt_chf(unit)
            table_data.append(
                [
                    Paragraph(xml_escape(label), normal),
                    Paragraph(xml_escape(qty_s), right_style),
                    Paragraph(xml_escape(unit_s), right_style),
                    Paragraph(_fmt_chf(amount), right_style),
                ]
            )
    else:
        table_data.append(
            [
                Paragraph("Prestations plateforme LIRIE", normal),
                Paragraph("", right_style),
                Paragraph("", right_style),
                Paragraph(_fmt_chf(subtotal), right_style),
            ]
        )

    col_w = [
        usable_width_pt * 0.52,
        usable_width_pt * 0.12,
        usable_width_pt * 0.18,
        usable_width_pt * 0.18,
    ]
    lines_table = Table(table_data, colWidths=col_w)
    lines_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.lightgrey),
            ]
        )
    )
    story.append(lines_table)
    story.append(Spacer(1, 12))

    tax_zero = Decimal(str(tax_rate or 0)) == 0
    tax_label = (
        "TVA non applicable (franchise)"
        if tax_zero
        else f"TVA {_fmt_tax_rate(tax_rate)} %"
    )
    totals = Table(
        [
            [
                Paragraph("<b>Sous-total HT</b>", right_style),
                Paragraph(f"{_fmt_chf(subtotal)} CHF", right_style),
            ],
            [
                Paragraph(f"<b>{tax_label}</b>", right_style),
                Paragraph(f"{_fmt_chf(tax_amount)} CHF", right_style),
            ],
            [
                Paragraph("<b>Total TTC</b>", right_style),
                Paragraph(f"<b>{_fmt_chf(total)} CHF</b>", right_style),
            ],
            [
                Paragraph("<b>Montant QR-facture</b>", right_style),
                Paragraph(f"<b>{_fmt_chf(qr_amount)} CHF</b>", right_style),
            ],
        ],
        colWidths=[usable_width_pt * 0.7, usable_width_pt * 0.3],
    )
    totals.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LINEABOVE", (0, -2), (-1, -2), 0.8, colors.black),
            ]
        )
    )
    story.append(totals)
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Paiement : QR-facture suisse officielle en page suivante.",
            label_style,
        )
    )

    # Page QR-Bill (paiement suisse) — jamais en mode aperçu
    # Le QR marketing LIRIE est dessiné en pied de page via onPage (hors flux).
    if not preview:
        story.append(NextPageTemplate("QRBill"))
        story.append(PageBreak())

        creditor_party = QrParty(
            name=str(creditor_snap.get("legal_name") or "LIRIE"),
            street=str(creditor_snap.get("street_name") or ""),
            building_number=creditor_snap.get("building_number"),
            postal_code=str(creditor_snap.get("postal_code") or ""),
            city=str(creditor_snap.get("city") or ""),
            country_code=str(creditor_snap.get("country_code") or "CH"),
        )
        debtor_party = QrParty(
            name=str(debtor_snap.get("legal_name") or "Entreprise"),
            street=str(debtor_snap.get("street_name") or ""),
            building_number=debtor_snap.get("building_number"),
            postal_code=str(debtor_snap.get("postal_code") or ""),
            city=str(debtor_snap.get("city") or ""),
            country_code=str(debtor_snap.get("country_code") or "CH"),
        )
        iban_clean = (iban or "").replace(" ", "")
        ref_type = resolve_platform_reference_mode(iban_clean, payment_reference_mode)
        payload = SwissQrBillPayload(
            creditor=creditor_party,
            debtor=debtor_party,
            iban=iban_clean,
            reference_type=ref_type,
            reference=qr_reference if ref_type in ("QRR", "SCOR") else None,
            amount=platform_qr_amount(qr_amount),
            currency="CHF",
            additional_information=invoice_number[:140],
        )
        try:
            rendered = render_swiss_qr_bill(payload)
            bill = rendered["qr_bill"]
            svg_buf = StringIO()
            bill.as_svg(svg_buf)
            drawing = _svg_content_to_drawing(
                svg_buf.getvalue().encode("utf-8"),
                inset_qrbill_text=True,
            )
            if drawing is not None:
                story.append(_make_qr_bill_flowable(drawing))
            else:
                story.append(Paragraph("QR-facture indisponible (rendu SVG).", normal))
        except Exception:
            logger.exception("Échec génération QR-Bill plateforme")
            story.append(
                Paragraph(
                    "QR-facture non générée — vérifiez IBAN/QR-IBAN et la référence.",
                    normal,
                )
            )

    doc.build(story)
    return buffer.getvalue()


def store_platform_invoice_pdf(
    invoice_number: str, pdf_bytes: bytes
) -> tuple[str, str]:
    """Écrit le PDF (legacy) — préférer publish_platform_invoice_pdf."""
    return publish_platform_invoice_pdf(invoice_number, pdf_bytes)


def publish_platform_invoice_pdf(
    invoice_number: str,
    pdf_bytes: bytes,
    *,
    previous_path: str | None = None,
) -> tuple[str, str]:
    """Publie un PDF sous clé immuable {number}_{checksum}.pdf.

    N'écrase pas l'ancien fichier : le pointeur DB est mis à jour par l'appelant
    après commit ; l'ancien fichier est supprimé en best-effort.
    """
    _PDF_ROOT.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256(pdf_bytes).hexdigest()
    safe_number = invoice_number.replace("/", "_").replace("\\", "_")
    filename = f"{safe_number}_{checksum[:16]}.pdf"
    path = _PDF_ROOT / filename
    if not path.exists():
        tmp = path.with_suffix(".pdf.tmp")
        tmp.write_bytes(pdf_bytes)
        tmp.replace(path)
    if previous_path and previous_path != str(path):
        try:
            old = Path(previous_path)
            if old.is_file() and old.resolve() != path.resolve():
                old.unlink(missing_ok=True)
        except OSError:
            pass
    return str(path), checksum
