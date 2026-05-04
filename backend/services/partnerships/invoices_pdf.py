# services/partner_invoice_pdf_service.py
"""Service pour générer les PDFs des factures partenaires.

Ce module génère des factures partenaires avec EXACTEMENT le même template
que les factures client/clinique :
- Même structure de document (BaseDocTemplate + PageTemplates)
- Même footer fixe (callback onPage)
- Même page QR-Bill dédiée (NextPageTemplate + PageBreak + Spacer)
- Même génération de référence (SCOR)

Architecture:
- Réutilise _make_invoice_doc_with_qrbill_page() pour le document
- Réutilise _make_legal_footer_page_callback() pour le footer fixe
- Réutilise _make_qr_bill_table() pour le QR-Bill
- Génère une référence SCOR comme les factures client
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from flask import current_app
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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

from models import CompanyBillingSettings
from models.booking_transfer import BookingTransfer
from models.partner_invoice import PartnerInvoice
from services.documents.pdf import (
    INVOICE_PAGE_LEFT_MARGIN_CM,
    INVOICE_PAGE_RIGHT_MARGIN_CM,
    QR_BILL_SPACER_PT,
    _build_default_legal_footer_html,
    _format_company_contact_footer_bar,
    _load_logo_ratio_safe,
    _make_invoice_doc_with_qrbill_page,
    _make_legal_footer_page_callback,
    _make_qr_bill_table,
    _svg_content_to_drawing,
    _xml_escape_for_paragraph,
)

# Constantes alignées avec pdf.py
DEST_ADDR_MAX_WIDTH_MM = 85.0  # Largeur max bloc destinataire
MONTHS_FR = (
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
# Constantes pour parsing d'adresse
MIN_ADDRESS_PARTS = 2
MIN_ADDRESS_PARTS_POSTAL = 3
MIN_ADDRESS_PARTS_CITY = 4
MONTHS_PER_YEAR = 12
MAX_CLIENT_NAME_LENGTH = 20

app_logger = logging.getLogger("partner_invoice_pdf_service")


def _escape_multiline_address_html(fragment: str) -> str:
    """Fragment pouvant contenir des ``<br/>`` (voir `_format_address_multiline`) — tout échapper."""
    import re

    if not fragment:
        return ""
    parts = re.split(r"(?i)<br\s*/?>", fragment)
    return "<br/>".join(_xml_escape_for_paragraph(p) for p in parts)


def _format_address_multiline(address: str | None) -> str:
    """Formate une adresse sur plusieurs lignes pour affichage PDF.

    Reproduit le comportement de pdf.py pour cohérence.
    """
    if not address:
        return "Adresse non renseignée"

    # Nettoyer et splitter
    clean = address.strip()
    parts = [p.strip() for p in clean.replace("\n", ",").split(",") if p.strip()]

    if len(parts) >= MIN_ADDRESS_PARTS_CITY:
        # Format: Rue, Numéro, CP, Ville
        return f"{parts[0]} {parts[1]}<br/>{parts[2]} {parts[3]}"
    if len(parts) >= MIN_ADDRESS_PARTS_POSTAL:
        # Format: Rue, CP, Ville
        return f"{parts[0]}<br/>{parts[1]} {parts[2]}"
    if len(parts) >= MIN_ADDRESS_PARTS:
        return f"{parts[0]}<br/>{parts[1]}"

    return clean


def _parse_address_for_qrbill(address: str | None) -> tuple[str, str, str]:
    """Parse une adresse pour QR-Bill (rue, code postal, ville).

    Identique à la logique de QRBillService.
    """
    if not address:
        return ("", "1200", "Genève")

    parts = [p.strip() for p in address.replace("\n", ",").split(",") if p.strip()]

    if len(parts) >= MIN_ADDRESS_PARTS_CITY:
        return (f"{parts[0]} {parts[1]}", parts[2], parts[3])
    if len(parts) >= MIN_ADDRESS_PARTS_POSTAL:
        return (parts[0], parts[1], parts[2])
    if len(parts) >= MIN_ADDRESS_PARTS:
        # Essayer d'extraire CP et ville de la dernière partie
        last = parts[-1].split()
        if len(last) >= MIN_ADDRESS_PARTS and last[0].isdigit():
            return (parts[0], last[0], " ".join(last[1:]))
        return (parts[0], parts[1], "Genève")

    return (address, "1200", "Genève")


def _generate_partner_scor_reference(partner_invoice: PartnerInvoice) -> str | None:
    """Génère une référence SCOR (ISO 11649) pour une facture partenaire.

    Format identique aux factures client/clinique pour cohérence.

    Args:
        partner_invoice: Facture partenaire

    Returns:
        Référence SCOR (RF...) ou None si erreur
    """
    try:
        from services.billing import generate_scor_reference

        # Utiliser le numéro de facture partenaire pour générer la référence SCOR
        # Le numéro est déjà préfixé "PARTNER-" donc unique
        return generate_scor_reference(
            partner_invoice.invoice_number,
            company_id=partner_invoice.executing_company_id,
        )
    except Exception as e:
        app_logger.warning(
            "Impossible de générer la référence SCOR pour facture partenaire %s: %s",
            partner_invoice.invoice_number,
            e,
        )
        return None


def generate_partner_invoice_pdf_content(
    partner_invoice: PartnerInvoice,
    transfers: list[BookingTransfer],
    *,
    line_amounts: dict[int, Any] | None = None,
) -> bytes:
    """Génère le contenu PDF d'une facture partenaire.

    Utilise EXACTEMENT le même template que les factures client/clinique:
    - BaseDocTemplate avec PageTemplates (First, Later, QRBill)
    - Footer fixe via callback onPage (pas dans le flow)
    - Page QR-Bill dédiée avec NextPageTemplate + PageBreak + Spacer
    - Référence SCOR générée comme pour client/clinique

    Args:
        partner_invoice: Facture partenaire
        transfers: Liste des transferts inclus dans la facture
        line_amounts: Montants par transfer_id (après overrides) pour cohérence ligne/total

    Returns:
        Contenu PDF en bytes
    """
    buffer = BytesIO()
    styles = getSampleStyleSheet()

    # === STYLES identiques à pdf.py ===
    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=4,
        fontName="Helvetica",
        leading=11,
    )

    centered_style = ParagraphStyle(
        "Centered",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#4a4a4a"),
        alignment=TA_CENTER,
        spaceAfter=4,
        fontName="Helvetica",
    )

    # === DÉTERMINER LES ENTREPRISES ===
    partnership = partner_invoice.partnership
    if not transfers:
        raise ValueError("Aucun transfert fourni pour la facture partenaire")

    executing_company = transfers[0].executing_company
    if executing_company.id == partnership.owner_company_id:
        billed_company = partnership.partner_company
    else:
        billed_company = partnership.owner_company

    if not executing_company:
        raise ValueError("Entreprise exécutante non trouvée")
    if not billed_company:
        raise ValueError("Entreprise destinataire non trouvée")

    # === CHARGEMENT DU LOGO (identique à pdf.py) ===
    logo_img = None
    logo_width = 0.0

    if hasattr(executing_company, "logo_url") and executing_company.logo_url:
        try:
            logo_url = executing_company.logo_url.strip()
            if not logo_url.startswith(("http://", "https://")):
                logo_url_clean = logo_url.lstrip("/")
                if logo_url_clean.startswith("uploads/"):
                    logo_url_clean = logo_url_clean[8:]
                uploads_dir = Path(
                    current_app.config.get("UPLOAD_FOLDER", "/app/uploads")
                )
                logo_path = uploads_dir / logo_url_clean
                if logo_path and Path(logo_path).exists():
                    max_width_pt = 595 * 0.24  # Même ratio que pdf.py
                    logo_img, logo_width, _ = _load_logo_ratio_safe(
                        logo_path, max_width_pt
                    )
        except Exception as e:
            app_logger.warning("Impossible de charger le logo: %s", e)

    # Récupérer billing_settings pour TVA et footer
    billing_settings = CompanyBillingSettings.query.filter_by(
        company_id=executing_company.id
    ).first()

    # === CONSTRUIRE LE FOOTER (identique à pdf.py) ===
    # Délai de paiement: priorité aux paramètres de l'entreprise exécutante.
    payment_terms_days = int(
        billing_settings.payment_terms_days
        if billing_settings and billing_settings.payment_terms_days
        else (
            partnership.payment_terms_days
            if partnership and partnership.payment_terms_days
            else 30
        )
    )

    overdue_fee = 5.00
    if billing_settings and billing_settings.overdue_fee:
        overdue_fee = float(billing_settings.overdue_fee)

    # IBAN depuis billing_settings
    iban_value = None
    if billing_settings and billing_settings.iban:
        iban_value = billing_settings.iban

    footer_message = _build_default_legal_footer_html(
        payment_terms_days, overdue_fee, iban_value
    )

    _emit_name = executing_company.name or "[Nom entreprise non configuré]"
    _emit_email = (
        executing_company.billing_email or executing_company.contact_email or ""
    )
    _emit_phone = executing_company.contact_phone or ""
    _emit_uid = executing_company.uid_ide or ""
    contact_bar = _format_company_contact_footer_bar(
        _emit_name, _emit_email, _emit_phone, _emit_uid
    )

    # Créer le callback footer (IDENTIQUE à pdf.py)
    footer_cb = _make_legal_footer_page_callback(
        footer_message,
        mention=None,  # Pas de mention spéciale pour les factures partenaires
        centered_style=centered_style,
        contact_bar=contact_bar,
    )

    # Callback pour la première page (footer + debug envelope si activé)
    def _on_first_page(canvas: Any, doc: Any) -> None:
        footer_cb(canvas, doc)

    # === CRÉER LE DOCUMENT avec PageTemplates (IDENTIQUE à pdf.py) ===
    doc = _make_invoice_doc_with_qrbill_page(
        buffer,
        top_margin_cm=2,
        bottom_margin_cm=2.5,  # Réserve espace pour pied de page légal
        left_margin_cm=INVOICE_PAGE_LEFT_MARGIN_CM,
        right_margin_cm=INVOICE_PAGE_RIGHT_MARGIN_CM,
        on_first_page=_on_first_page,
    )

    story: list[Any] = []

    # === EN-TÊTE : ENTREPRISE (gauche) | DESTINATAIRE (droite) ===
    # Structure identique à pdf.py

    # Informations entreprise émettrice
    company_name = executing_company.name or "[Nom entreprise non configuré]"
    company_address = _format_address_multiline(executing_company.address)
    company_phone = executing_company.contact_phone or ""
    company_email = (
        executing_company.billing_email or executing_company.contact_email or ""
    )
    company_uid = executing_company.uid_ide or ""

    # Statut TVA (identique à pdf.py)
    vat_status_text = ""
    if billing_settings and billing_settings.vat_applicable:
        vat_number = billing_settings.vat_number or ""
        if vat_number:
            vat_status_text = f"N° TVA : {vat_number}"
        else:
            vat_status_text = f"TVA {billing_settings.vat_rate or 7.7}% incluse"

    _addr_emit = _escape_multiline_address_html(company_address)
    company_info_html = (
        f"{_xml_escape_for_paragraph(company_name)}<br/>"
        f"{_addr_emit}"
    )
    if vat_status_text:
        company_info_html += f"<br/>{_xml_escape_for_paragraph(vat_status_text)}"
    company_para = Paragraph(company_info_html, normal_style)

    # Informations destinataire (partenaire)
    billed_name = billed_company.name or "Entreprise"
    billed_address = _format_address_multiline(billed_company.address)
    billed_email = billed_company.billing_email or billed_company.contact_email or ""
    billed_phone = billed_company.contact_phone or ""

    _bill_addr = _escape_multiline_address_html(billed_address)
    recipient_parts = [
        f"<b>{_xml_escape_for_paragraph(billed_name)}</b>",
        _bill_addr,
    ]
    if billed_email:
        recipient_parts.append(_xml_escape_for_paragraph(billed_email))
    if billed_phone:
        recipient_parts.append(_xml_escape_for_paragraph(billed_phone))
    recipient_html = "<br/>".join(recipient_parts)

    # Label "Facturé à" avec même style que pdf.py
    label_style = ParagraphStyle(
        "DestLabel",
        parent=normal_style,
        fontSize=8,
        spaceAfter=2,
    )
    label_para = Paragraph("<b>Facturé à :</b>", label_style)
    recipient_para = Paragraph(recipient_html, normal_style)

    dest_width_pt = DEST_ADDR_MAX_WIDTH_MM * mm
    recipient_block = Table(
        [[label_para], [recipient_para]],
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

    # Construire cellule gauche (logo + entreprise)
    page_width_pt = A4[0]
    usable_width_pt = (
        page_width_pt
        - INVOICE_PAGE_LEFT_MARGIN_CM * cm
        - INVOICE_PAGE_RIGHT_MARGIN_CM * cm
    )
    company_width_pt = usable_width_pt - dest_width_pt

    left_cell_content: list[Any] = []
    if logo_img:
        is_drawing = (
            hasattr(logo_img, "width")
            and hasattr(logo_img, "height")
            and hasattr(logo_img, "scale")
        )
        if is_drawing:
            logo_table = Table([[logo_img]], colWidths=[logo_width])
            logo_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            left_cell_content.append(logo_table)
        else:
            left_cell_content.append(logo_img)
            left_cell_content.append(Spacer(1, 8))
    left_cell_content.append(company_para)

    # Table d'en-tête à deux colonnes (identique à pdf.py)
    recipient_top_padding_mm = 25.0
    recipient_left_padding_mm = 15.0

    header_table = Table(
        [[left_cell_content, recipient_block]],
        colWidths=[company_width_pt, dest_width_pt],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, -1), 6),
                ("LEFTPADDING", (1, 0), (1, -1), recipient_left_padding_mm * mm),
                ("RIGHTPADDING", (1, 0), (1, -1), 0),
                ("TOPPADDING", (1, 0), (1, -1), recipient_top_padding_mm * mm),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 20))

    # === INFORMATIONS FACTURE (identique à pdf.py) ===
    # Période en texte français (comme pdf.py)
    period_month = partner_invoice.period_month
    period_year = partner_invoice.period_year
    if 1 <= period_month <= MONTHS_PER_YEAR:
        period_label = f"{MONTHS_FR[period_month - 1]} {period_year}"
    else:
        period_label = f"{period_month:02d}.{period_year}"

    _pinv = _xml_escape_for_paragraph(str(partner_invoice.invoice_number or ""))
    _per_esc = _xml_escape_for_paragraph(period_label)
    _iss = (
        partner_invoice.issued_at.strftime("%d.%m.%Y")
        if partner_invoice.issued_at
        else "N/A"
    )
    _due = (
        partner_invoice.due_date.strftime("%d.%m.%Y")
        if partner_invoice.due_date
        else "N/A"
    )
    invoice_info_html = (
        f"<b>Numéro de facture :</b> {_pinv}<br/>"
        f"<b>Date d'émission :</b> {_iss}<br/>"
        f"<b>Date d'échéance :</b> {_due}<br/>"
        f"<b>Période de facturation :</b> {_per_esc}"
    )
    story.append(Paragraph(invoice_info_html, normal_style))
    story.append(Spacer(1, 20))

    # === TABLEAU DES TRANSFERTS (style identique à pdf.py) ===
    def format_address_for_table(address: str | None, max_len: int = 30) -> str:
        if not address:
            return "N/A"
        clean = address.replace(", Suisse", "").replace(" Suisse", "").strip()
        if len(clean) <= max_len:
            return clean
        return clean[: max_len - 1] + "…"

    # En-tête du tableau
    table_data: list[list[Any]] = [
        ["Date", "Client", "Départ", "Arrivée", "Montant CHF"]
    ]

    for transfer in transfers:
        booking = transfer.booking
        if booking:
            date_str = (
                booking.scheduled_time.strftime("%d.%m.%Y")
                if booking.scheduled_time
                else ""
            )
            # Nom du client
            client_name = ""
            if booking.client and booking.client.user:
                client_name = (
                    booking.customer_name
                    or f"{booking.client.user.first_name or ''} {booking.client.user.last_name or ''}".strip()
                    or booking.client.user.username
                    or "Client"
                )
                if len(client_name) > MAX_CLIENT_NAME_LENGTH:
                    client_name = client_name[: MAX_CLIENT_NAME_LENGTH - 1] + "…"
            else:
                client_name = booking.customer_name or "Client"

            departure = format_address_for_table(booking.pickup_location)
            arrival = format_address_for_table(booking.dropoff_location)
        else:
            date_str = ""
            client_name = "N/A"
            departure = "N/A"
            arrival = "N/A"

        # Montant effectif (override si fourni, sinon partner_cost)
        line_amt = (line_amounts or {}).get(transfer.id)
        if line_amt is not None:
            amount = f"{float(line_amt):.2f}"
        elif transfer.partner_cost is not None:
            amount = f"{float(transfer.partner_cost):.2f}"
        else:
            amount = "0.00"
        table_data.append([date_str, client_name, departure, arrival, amount])

    # Style tableau IDENTIQUE à pdf.py (pas de couleurs de fond) ; largeur totale = zone utile
    _cols_scale = usable_width_pt / (17 * cm)
    services_table = Table(
        table_data,
        colWidths=[
            2 * cm * _cols_scale,
            3.5 * cm * _cols_scale,
            4.5 * cm * _cols_scale,
            4.5 * cm * _cols_scale,
            2.5 * cm * _cols_scale,
        ],
    )
    services_table.setStyle(
        TableStyle(
            [
                # En-tête
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("ALIGN", (-1, 0), (-1, 0), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                # Corps
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ALIGN", (0, 1), (-1, -1), "LEFT"),
                ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                ("TOPPADDING", (0, 1), (-1, -1), 6),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.lightgrey),
            ]
        )
    )

    story.append(services_table)
    story.append(Spacer(1, 15))

    # === TOTAL (style identique à pdf.py) ===
    total_amount = float(partner_invoice.total_amount)
    vat_amount = float(partner_invoice.vat_amount)
    subtotal_amount = float(partner_invoice.subtotal_amount)

    # Ligne de séparation
    total_separator = Table([[""]], colWidths=[usable_width_pt])
    total_separator.setStyle(
        TableStyle([("LINEBELOW", (0, 0), (0, 0), 1, colors.black)])
    )
    story.append(total_separator)
    story.append(Spacer(1, 8))

    # Tableau des totaux (aligné à droite comme pdf.py)
    if vat_amount > 0:
        total_data = [
            ["", "", "", "Sous-total HT :", f"{subtotal_amount:.2f}"],
            ["", "", "", "TVA :", f"{vat_amount:.2f}"],
            ["", "", "", "TOTAL :", f"{total_amount:.2f}"],
        ]
    else:
        total_data = [
            ["", "", "", f"Nombre de transferts : {len(transfers)}", ""],
            ["", "", "", "TOTAL :", f"{total_amount:.2f}"],
        ]

    total_table = Table(
        total_data,
        colWidths=[
            2 * cm * _cols_scale,
            3.5 * cm * _cols_scale,
            4.5 * cm * _cols_scale,
            4.5 * cm * _cols_scale,
            2.5 * cm * _cols_scale,
        ],
    )
    total_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ALIGN", (3, 0), (4, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("FONTNAME", (0, 0), (-1, -2), "Helvetica"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(total_table)

    # NOTE: Le footer est géré par le callback onPage, pas dans le flow
    # C'est IDENTIQUE à pdf.py - le footer est dessiné sur le canvas

    # === QR-BILL SUISSE OFFICIEL SUR PAGE SÉPARÉE (IDENTIQUE à pdf.py) ===
    # Forcer une nouvelle page avec le template QRBill (marge bas réduite)
    story.append(NextPageTemplate("QRBill"))
    story.append(PageBreak())

    # Espacement pour pousser le QR-Bill en bas de sa page (IDENTIQUE à pdf.py)
    story.append(Spacer(1, QR_BILL_SPACER_PT))

    try:
        if not billing_settings or not billing_settings.iban:
            story.append(
                Paragraph("QR-Bill non disponible - IBAN non configuré", normal_style)
            )
        else:
            # Générer la référence SCOR (comme pour client/clinique)
            scor_reference = _generate_partner_scor_reference(partner_invoice)

            app_logger.info(
                "Génération QR-Bill pour facture partenaire %s avec référence SCOR: %s",
                partner_invoice.invoice_number,
                scor_reference,
            )

            # Générer le QR-Bill avec la bibliothèque qrbill
            import tempfile

            from qrbill import QRBill

            creditor_street, creditor_pcode, creditor_city = _parse_address_for_qrbill(
                executing_company.address
            )
            if not creditor_street:
                creditor_street = "Adresse non renseignée"
                creditor_pcode = "1200"
                creditor_city = "Genève"

            debtor_street, debtor_pcode, debtor_city = _parse_address_for_qrbill(
                billed_company.address
            )
            if not debtor_street:
                debtor_street = "Adresse non renseignée"
                debtor_pcode = "1200"
                debtor_city = "Genève"

            qr_bill = QRBill(
                account=billing_settings.iban,
                creditor={
                    "name": executing_company.name or "Entreprise",
                    "street": creditor_street,
                    "pcode": creditor_pcode,
                    "city": creditor_city,
                    "country": "CH",
                },
                debtor={
                    "name": billed_company.name or "Entreprise",
                    "street": debtor_street,
                    "pcode": debtor_pcode,
                    "city": debtor_city,
                    "country": "CH",
                },
                amount=str(partner_invoice.total_amount),
                currency="CHF",
                # Référence SCOR (comme client/clinique)
                reference_number=scor_reference,
                additional_information=(
                    f"Facture {partner_invoice.invoice_number} - "
                    f"Période: {period_label}"
                ),
                language="fr",
            )

            # Générer le SVG
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".svg", delete=False
            ) as temp_svg:
                qr_bill.as_svg(temp_svg.name)
                with Path(temp_svg.name).open("r", encoding="utf-8") as f:
                    svg_content = f.read()
                Path(temp_svg.name).unlink()

            # Utiliser les mêmes helpers que pdf.py pour le rendu
            drawing = _svg_content_to_drawing(svg_content)
            if drawing:
                story.append(_make_qr_bill_table(drawing))
            else:
                story.append(Paragraph("QR-Bill non disponible", normal_style))

    except Exception as e:
        app_logger.warning("Impossible de générer le QR-Bill: %s", e)
        story.append(Paragraph("QR-Bill non disponible", normal_style))

    # Générer le PDF (callbacks dans PageTemplates - IDENTIQUE à pdf.py)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
