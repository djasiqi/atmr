# services/partner_invoice_pdf_service.py
"""Service pour générer les PDFs des factures partenaires.

Ce module génère des factures partenaires avec EXACTEMENT le même template
que les factures client/clinique :
- Même structure de document (BaseDocTemplate + PageTemplates)
- Même footer fixe (callback onPage)
- Même page QR-Bill dédiée (NextPageTemplate + PageBreak)
- Même génération de référence (SCOR)

Architecture:
- Réutilise _make_invoice_doc_with_qrbill_page() pour le document
- Réutilise _make_legal_footer_page_callback() pour le footer fixe
- Réutilise _make_qr_bill_flowable() pour le QR-Bill
- Génère une référence SCOR comme les factures client
"""

import logging
from decimal import Decimal
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
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from models import CompanyBillingSettings
from models.booking_transfer import BookingTransfer
from models.partner_invoice import PartnerInvoice
from services.documents.pdf import (
    INVOICE_PAGE_BOTTOM_MARGIN_FIRST_CM,
    INVOICE_PAGE_BOTTOM_MARGIN_LATER_CM,
    INVOICE_PAGE_LEFT_MARGIN_CM,
    INVOICE_PAGE_RIGHT_MARGIN_CM,
    INVOICE_PAGE_TOP_MARGIN_CM,
    _build_default_legal_footer_html,
    _format_company_contact_footer_bar,
    _load_logo_ratio_safe,
    _make_invoice_doc_with_qrbill_page,
    _make_legal_footer_page_callback,
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


def _partner_vat_header(billing_settings: Any, partner_invoice: PartnerInvoice) -> str:
    """Libellé d'en-tête aligné sur les montants enregistrés.

    ``calculate_vat`` ajoute la TVA au hors taxe puis arrondit le total.
    « incluse » n'est utilisé que si le hors taxe est déjà le total.
    Le taux 7.7 par défaut du modèle n'est pas réécrit ici.
    """
    number = ""
    rate = None
    applicable = False
    if billing_settings is not None:
        number = str(getattr(billing_settings, "vat_number", None) or "").strip()
        rate = getattr(billing_settings, "vat_rate", None)
        applicable = bool(getattr(billing_settings, "vat_applicable", False))
    if number:
        return f"N° TVA : {number}"
    if not applicable:
        return "TVA non applicable"
    if rate is None:
        return "TVA — taux non documenté"
    subtotal = Decimal(str(partner_invoice.subtotal_amount or 0)).quantize(
        Decimal("0.01")
    )
    vat_amount = Decimal(str(partner_invoice.vat_amount or 0)).quantize(Decimal("0.01"))
    total = Decimal(str(partner_invoice.total_amount or 0)).quantize(Decimal("0.01"))
    if vat_amount == 0:
        return "TVA non applicable"
    if subtotal + vat_amount == total:
        return f"TVA {rate}% en sus"
    if subtotal == total:
        return f"TVA {rate}% incluse"
    return f"TVA {rate}% — montants incohérents"


def _generate_partner_scor_reference(partner_invoice: PartnerInvoice) -> str | None:
    """Génère une référence SCOR (ISO 11649) pour une facture partenaire.

    Format identique aux factures client/clinique pour cohérence.

    Args:
        partner_invoice: Facture partenaire

    Returns:
        Référence SCOR (RF...) ou None si erreur
    """
    try:
        from services.billing.payment_reference_generator import ensure_partner_scor

        return ensure_partner_scor(partner_invoice)
    except Exception as e:
        app_logger.warning(
            "Référence SCOR partenaire refusée pour %s: %s",
            partner_invoice.invoice_number,
            e,
        )
        return None


def _route_from_partner_snapshot(snap: dict[str, Any]) -> tuple[str, str]:
    """Départ et arrivée du snapshot, sans relire la course."""
    departure = str(snap.get("departure") or "").strip()
    arrival = str(snap.get("arrival") or "").strip()
    if departure or arrival:
        return departure, arrival
    description = str(snap.get("description") or "")
    if " → " not in description:
        return "", ""
    left, arrival = description.split(" → ", 1)
    if " — " in left:
        _client, departure = left.split(" — ", 1)
    else:
        departure = left
    return departure.strip(), arrival.strip()


def generate_partner_invoice_pdf_content(
    partner_invoice: PartnerInvoice,
    transfers: list[BookingTransfer],
    *,
    line_amounts: dict[int, Any] | None = None,
    line_snapshots: list[dict[str, Any]] | None = None,
) -> bytes:
    """Génère le contenu PDF d'une facture partenaire.

    Utilise EXACTEMENT le même template que les factures client/clinique:
    - BaseDocTemplate avec PageTemplates (First, Later, QRBill)
    - Footer fixe via callback onPage (pas dans le flow)
    - Page QR-Bill dédiée avec NextPageTemplate + PageBreak
    - Référence SCOR générée comme pour client/clinique

    Args:
        partner_invoice: Facture partenaire
        transfers: Liste des transferts inclus dans la facture
        line_amounts: Montants par transfer_id (après overrides) pour cohérence ligne/total
        line_snapshots: Lignes persistées (libellé / montant édités). Prioritaires sur les transferts live.

    Returns:
        Contenu PDF en bytes
    """
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    from services.documents.pdf import (
        FONT_BODY,
        FONT_SECONDARY,
        _ensure_dejavu_pdf_fonts,
    )

    font_name, _font_name_bold = _ensure_dejavu_pdf_fonts()
    _body_leading = round(FONT_BODY * 1.3)
    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=FONT_BODY,
        leading=_body_leading,
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=6,
        fontName=font_name,
        leftIndent=0,
        rightIndent=0,
        firstLineIndent=0,
    )
    centered_style = ParagraphStyle(
        "Centered",
        parent=styles["Normal"],
        fontSize=FONT_BODY,
        leading=_body_leading,
        textColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=6,
        fontName=font_name,
    )

    # === DÉTERMINER LES ENTREPRISES ===
    from models.company import Company

    partnership = partner_invoice.partnership
    if not transfers and not line_snapshots:
        raise ValueError("Aucun transfert fourni pour la facture partenaire")

    executing_company = transfers[0].executing_company if transfers else None
    if executing_company is None:
        executing_company = Company.query.get(partner_invoice.executing_company_id)
    if partnership is None:
        raise ValueError("Partenariat introuvable pour la facture partenaire")
    if (
        executing_company is not None
        and executing_company.id == partnership.owner_company_id
    ):
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
        top_margin_cm=INVOICE_PAGE_TOP_MARGIN_CM,
        bottom_margin_cm=INVOICE_PAGE_BOTTOM_MARGIN_FIRST_CM,
        left_margin_cm=INVOICE_PAGE_LEFT_MARGIN_CM,
        right_margin_cm=INVOICE_PAGE_RIGHT_MARGIN_CM,
        on_first_page=_on_first_page,
        bottom_margin_later_cm=INVOICE_PAGE_BOTTOM_MARGIN_LATER_CM,
    )

    story: list[Any] = []

    # === EN-TÊTE : ENTREPRISE (gauche) | DESTINATAIRE (droite) ===
    # Structure identique à pdf.py

    # Informations entreprise émettrice
    company_name = executing_company.name or "[Nom entreprise non configuré]"
    company_address = _format_address_multiline(executing_company.address)

    vat_status_text = _partner_vat_header(billing_settings, partner_invoice)

    from services.documents.invoice_pdf_presentation import issuer_block_html

    _addr_emit = _escape_multiline_address_html(company_address)
    company_para = Paragraph(
        issuer_block_html(company_name, _addr_emit, vat_status_text),
        normal_style,
    )

    # Informations destinataire (partenaire) — overrides snapshot si présents
    billed_name = (
        partner_invoice.recipient_name
        or (billed_company.name if billed_company else None)
        or "Entreprise"
    )
    billed_address = partner_invoice.recipient_address or _format_address_multiline(
        billed_company.address if billed_company else None
    )
    billed_email = ""
    billed_phone = ""
    if partner_invoice.recipient_contact:
        billed_email = partner_invoice.recipient_contact
    elif billed_company:
        billed_email = (
            billed_company.billing_email or billed_company.contact_email or ""
        )
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
        fontSize=FONT_SECONDARY,
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

    from services.documents.invoice_pdf_presentation import invoice_meta_html

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
    story.append(
        Paragraph(
            invoice_meta_html(
                invoice_number=str(partner_invoice.invoice_number or ""),
                issued_on=_iss,
                due_on=_due,
                period_label=period_label,
            ),
            normal_style,
        )
    )
    story.append(Spacer(1, 14))

    from services.documents.pdf import _detail_lines_heading_paragraph

    story.append(_detail_lines_heading_paragraph(styles, _font_name_bold))

    # === DÉTAIL DES PRESTATIONS (même tableau que client / institution) ===
    from application.invoices.partner_invoice_lines import describe_transfer_line
    from application.invoices.partner_pickup_time import pickup_display_label
    from services.documents.invoice_pdf_columns import (
        LINE_TIME_PICKUP,
        normalize_line_time_mode,
    )
    from services.documents.invoice_pdf_presentation import (
        build_prestation_detail_table,
    )

    line_time_mode = normalize_line_time_mode(
        getattr(partner_invoice, "line_time_mode", None)
    )
    service_rows: list[dict[str, Any]] = []
    if line_snapshots:
        for snap in line_snapshots:
            amount_value = snap.get("amount")
            if amount_value is None:
                amount_value = snap.get("line_total") or 0
            departure, arrival = _route_from_partner_snapshot(snap)
            raw_meta = snap.get("line_meta")
            line_meta = raw_meta if isinstance(raw_meta, dict) else {}
            catalog_amount = None
            if line_meta.get("global_discount_applied"):
                initial = line_meta.get("original_line_total")
                if initial not in (None, ""):
                    amount_value = initial
            else:
                catalog_amount = line_meta.get("original_line_total")
            service_rows.append(
                {
                    "date": str(snap.get("service_date") or ""),
                    "client_name": str(
                        snap.get("client_name") or snap.get("description") or "Client"
                    ),
                    "departure": departure,
                    "arrival": arrival,
                    "amount": amount_value,
                    "catalog_amount": catalog_amount,
                    "pickup": str(
                        snap.get("pickup_label")
                        or pickup_display_label(
                            snap.get("pickup_time_source"),
                            snap.get("scheduled_pickup_at"),
                            snap.get("boarded_at"),
                        )
                    ),
                }
            )
    else:
        for index, transfer in enumerate(transfers):
            line_amt = (line_amounts or {}).get(transfer.id)
            if line_amt is not None:
                amount = float(line_amt)
            elif transfer.partner_cost is not None:
                amount = float(transfer.partner_cost)
            else:
                amount = 0.0
            described = describe_transfer_line(
                transfer, amount=Decimal(str(amount)), sort_order=index
            )
            service_rows.append(
                {
                    "date": str(described.get("service_date") or ""),
                    "client_name": str(described.get("client_name") or "Client"),
                    "departure": str(described.get("departure") or ""),
                    "arrival": str(described.get("arrival") or ""),
                    "amount": amount,
                    "pickup": str(
                        pickup_display_label(
                            described.get("pickup_time_source"),
                            described.get("scheduled_pickup_at"),
                            described.get("boarded_at"),
                        )
                    ),
                }
            )

    story.append(
        build_prestation_detail_table(
            service_rows,
            show_pickup=line_time_mode == LINE_TIME_PICKUP,
            available_width_pt=usable_width_pt,
        )
    )
    story.append(Spacer(1, 12))

    from application.invoices.partner_invoice_lines import partner_invoice_editor_meta
    from services.documents.invoice_pdf_presentation import GRAND_TOTAL_LABEL
    from services.documents.pdf import (
        _build_totals_table,
        _extend_totals_flowables,
        _global_discount_hint_flowable,
    )

    editor_meta = partner_invoice_editor_meta(partner_invoice)
    hint = _global_discount_hint_flowable(
        type("_PartnerHint", (), {"meta": editor_meta})(),
        styles,
        font_name,
        content_width_pt=usable_width_pt,
    )
    if hint is not None:
        story.append(Spacer(1, 10))
        story.append(hint)
        story.append(Spacer(1, 9))

    totals_source = type(
        "_PartnerTotals",
        (),
        {
            "subtotal_amount": partner_invoice.subtotal_amount,
            "vat_total_amount": partner_invoice.vat_amount,
            "total_amount": partner_invoice.total_amount,
            "meta": editor_meta,
            "lines": [],
        },
    )()
    _extend_totals_flowables(
        story,
        _build_totals_table(
            totals_source,
            False,
            False,
            font_name,
            _font_name_bold,
            grand_total_label=GRAND_TOTAL_LABEL,
        ),
    )

    # NOTE: Le footer est géré par le callback onPage, pas dans le flow
    # C'est IDENTIQUE à pdf.py - le footer est dessiné sur le canvas

    # === QR-BILL SUISSE OFFICIEL SUR PAGE SÉPARÉE ===
    # La page n'existe que si le bulletin peut être dessiné.
    qr_drawing = None
    try:
        if billing_settings and billing_settings.iban:
            # Générer la référence SCOR (comme pour client/clinique)
            scor_reference = _generate_partner_scor_reference(partner_invoice)
            if not scor_reference:
                raise ValueError("reference_scor_absente")

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
            qr_drawing = _svg_content_to_drawing(svg_content, inset_qrbill_text=True)

    except Exception as e:
        app_logger.warning("Impossible de générer le QR-Bill: %s", e)
        qr_drawing = None
    from services.documents.pdf import append_qr_bill_page

    append_qr_bill_page(story, qr_drawing)

    # Générer le PDF (callbacks dans PageTemplates - IDENTIQUE à pdf.py)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
