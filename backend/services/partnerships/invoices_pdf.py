# services/partner_invoice_pdf_service.py
"""Service pour générer les PDFs des factures partenaires.

Ce module génère des factures partenaires avec le même design que les factures
client/clinique : logo, mise en page professionnelle, QR-Bill intégré.
"""

import logging
import tempfile
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Any

from flask import current_app
from qrbill import QRBill
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from models import CompanyBillingSettings
from models.booking_transfer import BookingTransfer
from models.partner_invoice import PartnerInvoice
from services.documents.pdf import (
    QR_BILL_SPACER_PT,
    _load_logo_ratio_safe,
    _make_qr_bill_table,
    _svg_content_to_drawing,
)

# Constantes pour éviter les valeurs magiques
MIN_ADDRESS_PARTS = 2
MIN_ADDRESS_PARTS_POSTAL = 3
MIN_ADDRESS_PARTS_CITY = 4
MIN_IBAN_LENGTH = 21  # Longueur minimale pour formater un IBAN

app_logger = logging.getLogger("partner_invoice_pdf_service")


def _format_address_for_display(address: str) -> str:
    """Formate une adresse pour l'affichage dans le PDF.

    Format de sortie :
    - Ligne 1 : Rue et numéro
    - Ligne 2 : Code postal et ville

    Args:
        address: Adresse complète (peut être au format "Rue, Numéro, Code Postal, Ville"
                 ou "Rue Numéro, Code Postal Ville" ou autres formats)

    Returns:
        Adresse formatée avec <br/> pour les retours à la ligne
    """
    if not address or address == "Adresse non renseignée":
        return "Adresse non renseignée"

    # Nettoyer l'adresse
    clean_address = address.strip()

    # Essayer différents formats d'adresse
    # Format 1: "Rue, Numéro, Code Postal, Ville" (avec virgules)
    parts = [p.strip() for p in clean_address.split(",")]

    result = None
    if len(parts) >= MIN_ADDRESS_PARTS_CITY:
        # Format: "Rue, Numéro, Code Postal, Ville"
        street_and_number = f"{parts[0]}, {parts[1]}"
        postal_code = parts[2]
        city = parts[3]
        result = f"{street_and_number}<br/>{postal_code} {city}"
    elif len(parts) >= MIN_ADDRESS_PARTS_POSTAL:
        # Format: "Rue Numéro, Code Postal, Ville" ou "Rue, Code Postal, Ville"
        street = parts[0]
        postal_code = parts[1]
        city = parts[2]
        result = f"{street}<br/>{postal_code} {city}"
    elif len(parts) >= MIN_ADDRESS_PARTS:
        # Format: "Rue Numéro, Code Postal Ville"
        street = parts[0]
        # Essayer d'extraire code postal et ville de la dernière partie
        last_part = parts[-1].strip()
        parts_space = last_part.split()
        if len(parts_space) >= MIN_ADDRESS_PARTS:
            postal_code = parts_space[0]
            city = " ".join(parts_space[1:])
            result = f"{street}<br/>{postal_code} {city}"
        else:
            # Si on ne peut pas parser, retourner tel quel avec un <br/> au milieu
            result = f"{street}<br/>{last_part}"

    if result:
        return result

    # Si le format n'est pas reconnu, essayer de trouver un code postal (4 chiffres)
    import re

    postal_match = re.search(r"\b(\d{4})\b", clean_address)
    if postal_match:
        postal_code = postal_match.group(1)
        postal_pos = clean_address.find(postal_code)
        street = clean_address[:postal_pos].strip().rstrip(",")
        city = clean_address[postal_pos + len(postal_code) :].strip()
        if street and city:
            return f"{street}<br/>{postal_code} {city}"

    # Fallback : retourner l'adresse telle quelle
    return clean_address


def _parse_address_for_qrbill(address: str) -> tuple[str, str, str]:
    """Parse une adresse pour QR-bill en séparant rue, code postal et ville.

    Format attendu: "Rue, Numéro, Code Postal, Ville" ou "Rue Numéro, Code Postal Ville"

    Args:
        address: Adresse complète

    Returns:
        Tuple (street, pcode, city)
    """
    if not address:
        return ("", "1200", "Genève")

    # Essayer format avec virgules: "Route de Chevrens, 145, 1247, Anières"
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= MIN_ADDRESS_PARTS_CITY:
        # Format: "Rue, Numéro, Code Postal, Ville"
        street = f"{parts[0]}, {parts[1]}"
        pcode = parts[2]
        city = parts[3]
        return (street, pcode, city)
    if len(parts) >= MIN_ADDRESS_PARTS_POSTAL:
        # Format: "Rue, Code Postal, Ville" ou "Rue Numéro, Code Postal, Ville"
        street = parts[0]
        pcode = parts[1]
        city = parts[2]
        return (street, pcode, city)
    if len(parts) >= MIN_ADDRESS_PARTS:
        # Format: "Rue Numéro, Code Postal Ville"
        street = parts[0]
        # Essayer d'extraire code postal et ville de la dernière partie
        last_part = parts[-1].strip()
        parts_space = last_part.split()
        if len(parts_space) >= MIN_ADDRESS_PARTS:
            pcode = parts_space[0]
            city = " ".join(parts_space[1:])
            return (street, pcode, city)

    # Fallback: utiliser l'adresse complète comme rue
    return (address, "1200", "Genève")


def generate_partner_invoice_pdf_content(
    partner_invoice: PartnerInvoice, transfers: list[BookingTransfer]
) -> bytes:
    """Génère le contenu PDF d'une facture partenaire.

    Utilise le même design que les factures client/clinique :
    - Logo de l'entreprise en haut à gauche
    - En-tête avec émetteur (gauche) et destinataire (droite)
    - Tableau des transferts
    - QR-Bill sur page séparée

    Args:
        partner_invoice: Facture partenaire
        transfers: Liste des transferts inclus dans la facture

    Returns:
        Contenu PDF en bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()

    # Styles cohérents avec les factures client/clinique
    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=4,
        fontName="Helvetica",
        leading=12,
    )

    right_style = ParagraphStyle(
        "Right",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.black,
        alignment=TA_RIGHT,
        spaceAfter=4,
        fontName="Helvetica",
        leading=12,
    )

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=colors.HexColor("#1a365d"),
        alignment=TA_LEFT,
        spaceAfter=10,
        spaceBefore=15,
        fontName="Helvetica-Bold",
    )

    story = []

    # === DÉTERMINER LES ENTREPRISES ===
    partnership = partner_invoice.partnership
    if not transfers:
        raise ValueError("Aucun transfert fourni pour la facture partenaire")

    # L'entreprise émettrice est celle qui exécute les courses (celle qui facture)
    executing_company = transfers[0].executing_company
    # L'entreprise destinataire est celle qui doit payer (l'autre entreprise du partenariat)
    if executing_company.id == partnership.owner_company_id:
        billed_company = partnership.partner_company
    else:
        billed_company = partnership.owner_company

    if not executing_company:
        raise ValueError("Entreprise exécutante non trouvée")
    if not billed_company:
        raise ValueError("Entreprise destinataire non trouvée")

    # === CHARGEMENT DU LOGO ===
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
                    max_width_pt = 595 * 0.22  # ~22% de la largeur A4
                    logo_img, logo_width, _ = _load_logo_ratio_safe(
                        logo_path, max_width_pt
                    )
        except Exception as e:
            app_logger.warning("Impossible de charger le logo: %s", e)

    # === EN-TÊTE : LOGO + ENTREPRISE (gauche) | DESTINATAIRE (droite) ===
    company_name = executing_company.name or "Entreprise"
    company_address_raw = executing_company.address or "Adresse non renseignée"
    company_address = _format_address_for_display(company_address_raw)
    company_phone = executing_company.contact_phone or ""
    company_email = (
        executing_company.billing_email
        or executing_company.contact_email
        or ""
    )
    company_uid = executing_company.uid_ide or ""

    # Construire le bloc émetteur
    company_info_parts = [f"<b>{company_name}</b>", company_address]
    if company_email:
        company_info_parts.append(f"Email : {company_email}")
    if company_phone:
        company_info_parts.append(f"Tél : {company_phone}")
    if company_uid:
        company_info_parts.append(f"IDE : {company_uid}")

    company_info_html = "<br/>".join(company_info_parts)

    # Construire le bloc destinataire
    billed_company_name = billed_company.name or "Entreprise"
    billed_company_address_raw = billed_company.address or "Adresse non renseignée"
    billed_company_address = _format_address_for_display(billed_company_address_raw)
    billed_company_phone = billed_company.contact_phone or ""
    billed_company_email = (
        billed_company.billing_email or billed_company.contact_email or ""
    )

    billed_info_parts = [
        "<b>Facturé à :</b>",
        f"<b>{billed_company_name}</b>",
        billed_company_address,
    ]
    if billed_company_email:
        billed_info_parts.append(f"Email : {billed_company_email}")
    if billed_company_phone:
        billed_info_parts.append(f"Tél : {billed_company_phone}")

    billed_info_html = "<br/>".join(billed_info_parts)

    # Créer le contenu de la cellule gauche (logo + infos entreprise)
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
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            left_cell_content.append(logo_table)
        else:
            left_cell_content.append(logo_img)
            left_cell_content.append(Spacer(1, 8))

    left_cell_content.append(Paragraph(company_info_html, normal_style))

    # Cellule droite (destinataire)
    right_cell_content = Paragraph(billed_info_html, right_style)

    # Table d'en-tête à deux colonnes
    header_table = Table(
        [[left_cell_content, right_cell_content]],
        colWidths=[9 * cm, 8 * cm],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (0, 0), "LEFT"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 25))

    # === TITRE DE LA FACTURE ===
    story.append(
        Paragraph(
            f"FACTURE PARTENAIRE N° {partner_invoice.invoice_number}",
            title_style,
        )
    )

    # === INFORMATIONS FACTURE ===
    invoice_info = f"""
    <b>Date d'émission :</b> {partner_invoice.issued_at.strftime("%d.%m.%Y") if partner_invoice.issued_at else "N/A"}<br/>
    <b>Date d'échéance :</b> {partner_invoice.due_date.strftime("%d.%m.%Y") if partner_invoice.due_date else "N/A"}<br/>
    <b>Période :</b> {partner_invoice.period_month:02d}/{partner_invoice.period_year}
    """
    story.append(Paragraph(invoice_info, normal_style))
    story.append(Spacer(1, 15))

    # === TABLEAU DES TRANSFERTS ===
    # Fonction pour formater les adresses
    def format_address_for_table(address, max_length=28):
        if not address or address == "Adresse inconnue":
            return "Adresse non renseignée"

        clean_address = address.replace(", Suisse", "").strip()
        import re

        clean_address = re.sub(r"^Trajet\s+", "", clean_address)
        clean_address = clean_address.replace(" Suisse", "").strip()
        clean_address = clean_address.replace(" · ", " ").replace("·", "")

        if len(clean_address) <= max_length:
            return clean_address

        words = clean_address.split(" ")
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= max_length:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return "\n".join(lines[:2])

    # Constante pour la longueur maximale du nom du client
    MAX_CLIENT_NAME_LENGTH = 20

    # Données du tableau (typage explicite pour éviter les erreurs pyright)
    table_data: list[list[Any]] = [
        ["Date", "Client", "Départ", "Arrivée", "Montant"]
    ]

    for transfer in transfers:
        booking = transfer.booking
        if booking:
            date_str = (
                booking.scheduled_time.strftime("%d.%m.%Y")
                if booking.scheduled_time
                else ""
            )
            # Récupérer le nom du client
            client_name = ""
            if booking.client and booking.client.user:
                client_name = (
                    booking.customer_name
                    or (
                        f"{booking.client.user.first_name or ''} "
                        f"{booking.client.user.last_name or ''}"
                    ).strip()
                    or booking.client.user.username
                    or "Client"
                )
                # Tronquer si trop long
                if len(client_name) > MAX_CLIENT_NAME_LENGTH:
                    client_name = client_name[: MAX_CLIENT_NAME_LENGTH - 1] + "…"
            else:
                client_name = booking.customer_name or "Client"

            departure = format_address_for_table(
                booking.pickup_location or "N/A", max_length=28
            )
            arrival = format_address_for_table(
                booking.dropoff_location or "N/A", max_length=28
            )
        else:
            date_str = ""
            client_name = "N/A"
            departure = "N/A"
            arrival = "N/A"

        amount = f"{transfer.partner_cost:.2f}" if transfer.partner_cost else "0.00"
        table_data.append([date_str, client_name, departure, arrival, amount])

    # Colonnes ajustées pour une meilleure lisibilité
    services_table = Table(
        table_data, colWidths=[2.2 * cm, 3 * cm, 4.5 * cm, 4.5 * cm, 2.3 * cm]
    )
    services_table.setStyle(
        TableStyle(
            [
                # En-tête avec fond coloré
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("ALIGN", (4, 0), (4, 0), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                # Corps du tableau
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ALIGN", (0, 1), (-1, -1), "LEFT"),
                ("ALIGN", (4, 1), (4, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                ("TOPPADDING", (0, 1), (-1, -1), 6),
                # Lignes alternées pour lisibilité
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                # Bordures
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#2c5282")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
            ]
        )
    )

    story.append(services_table)
    story.append(Spacer(1, 20))

    # === TOTAL ===
    subtotal_amount = float(partner_invoice.subtotal_amount)
    vat_amount = float(partner_invoice.vat_amount)
    total_amount = float(partner_invoice.total_amount)

    # Récupérer les paramètres de facturation
    billing_settings = CompanyBillingSettings.query.filter_by(
        company_id=executing_company.id
    ).first()

    vat_is_applicable = vat_amount > 0

    # Tableau du total aligné à droite avec design moderne
    if vat_is_applicable:
        total_data = [
            ["Sous-total HT :", f"CHF {subtotal_amount:.2f}"],
            ["TVA :", f"CHF {vat_amount:.2f}"],
            ["TOTAL TTC :", f"CHF {total_amount:.2f}"],
        ]
    else:
        total_data = [
            ["Nombre de transferts :", f"{len(transfers)}"],
            ["TOTAL À PAYER :", f"CHF {total_amount:.2f}"],
        ]

    total_table = Table(
        total_data, colWidths=[4 * cm, 3 * cm]
    )

    style_rules = [
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("FONTNAME", (0, 0), (-1, -2), "Helvetica"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, -1), (-1, -1), 11),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#edf2f7")),
        ("BOX", (0, -1), (-1, -1), 0.5, colors.HexColor("#2c5282")),
    ]
    total_table.setStyle(TableStyle(style_rules))

    # Wrapper pour aligner le tableau à droite
    total_wrapper = Table(
        [[total_table]],
        colWidths=[17 * cm],
    )
    total_wrapper.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (0, 0), "RIGHT"),
                ("VALIGN", (0, 0), (0, 0), "TOP"),
            ]
        )
    )

    story.append(total_wrapper)
    story.append(Spacer(1, 25))

    # === PIED DE PAGE ===
    payment_terms_days = 30
    if partnership and partnership.payment_terms_days:
        payment_terms_days = partnership.payment_terms_days
    elif billing_settings and billing_settings.payment_terms_days:
        payment_terms_days = int(billing_settings.payment_terms_days)

    overdue_fee = Decimal("15.00")
    if billing_settings and billing_settings.overdue_fee:
        overdue_fee = billing_settings.overdue_fee

    jours_text = "jours" if payment_terms_days > 1 else "jour"

    # Pas de fallback IBAN hardcodé pour éviter les paiements sur le mauvais compte
    iban_value = None
    if billing_settings and billing_settings.iban:
        iban_value = billing_settings.iban
    elif hasattr(executing_company, "iban") and executing_company.iban:
        iban_value = executing_company.iban

    # Style pour le footer
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#4a5568"),
        alignment=TA_CENTER,
        spaceAfter=4,
        fontName="Helvetica",
        leading=11,
    )

    # Message de paiement
    footer_message = (
        f"En votre aimable règlement net sous <b>{payment_terms_days} {jours_text}</b> "
        f"avec nos remerciements anticipés.<br/>"
        f"En cas de retard de paiement, des frais de rappel de CHF {overdue_fee:.2f} "
        f"seront facturés conformément à nos conditions générales."
    )

    # Séparateur visuel
    separator = Table([[""]], colWidths=[17 * cm])
    separator.setStyle(
        TableStyle([("LINEBELOW", (0, 0), (0, 0), 0.5, colors.HexColor("#cbd5e0"))])
    )
    story.append(separator)
    story.append(Spacer(1, 15))

    story.append(Paragraph(footer_message, footer_style))

    # Coordonnées bancaires si disponibles
    if iban_value:
        # Formater l'IBAN pour une meilleure lisibilité
        iban_formatted = iban_value
        if len(iban_value) >= MIN_IBAN_LENGTH and " " not in iban_value:
            iban_formatted = " ".join(
                [iban_value[i : i + 4] for i in range(0, len(iban_value), 4)]
            )
        bank_info = f"<b>Coordonnées bancaires :</b> IBAN {iban_formatted}"
        story.append(Spacer(1, 8))
        story.append(Paragraph(bank_info, footer_style))

    # === QR-BILL SUISSE OFFICIEL SUR PAGE SÉPARÉE ===
    story.append(PageBreak())
    story.append(Spacer(1, QR_BILL_SPACER_PT))

    try:
        # Récupérer les paramètres de facturation de l'entreprise émettrice
        billing_settings = CompanyBillingSettings.query.filter_by(
            company_id=executing_company.id
        ).first()

        if not billing_settings or not billing_settings.iban:
            story.append(
                Paragraph("QR-Bill non disponible - IBAN non configuré", normal_style)
            )
        else:
            # Parser les adresses correctement pour QR-bill (street, pcode, city séparés)
            creditor_street, creditor_pcode, creditor_city = _parse_address_for_qrbill(
                executing_company.address or ""
            )
            if not creditor_street:
                creditor_street = "Route de Chevrens, 145"
                creditor_pcode = "1247"
                creditor_city = "Anières"

            debtor_street, debtor_pcode, debtor_city = _parse_address_for_qrbill(
                billed_company.address or ""
            )
            if not debtor_street:
                debtor_street = "Adresse non renseignée"
                debtor_pcode = "1200"
                debtor_city = "Genève"

            # Créer le QR-Bill directement avec qrbill
            qr_bill = QRBill(
                account=billing_settings.iban,
                creditor={
                    "name": executing_company.name or "Emmenez Moi",
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
                reference_number=None,
                additional_information=(
                    f"Facture {partner_invoice.invoice_number} - "
                    f"Période: {partner_invoice.period_month:02d}.{partner_invoice.period_year}"
                ),
                language="fr",
            )

            # Générer le SVG du QR-Bill
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".svg", delete=False
            ) as temp_svg:
                qr_bill.as_svg(temp_svg.name)

                # Lire le contenu SVG
                with Path(temp_svg.name).open("r", encoding="utf-8") as f:
                    svg_content = f.read()

                # Nettoyer le fichier temporaire
                Path(temp_svg.name).unlink()

            # Convertir SVG en drawing ReportLab (svg2rlg n'accepte que path, pas BytesIO)
            drawing = _svg_content_to_drawing(svg_content)

            if drawing:
                story.append(_make_qr_bill_table(drawing))
            else:
                story.append(Paragraph("QR-Bill non disponible", normal_style))
    except Exception as e:
        app_logger.warning("Impossible de générer le QR-Bill: %s", e)
        story.append(Paragraph("QR-Bill non disponible", normal_style))

    # Générer le PDF
    doc.build(story)

    # Retourner le contenu
    buffer.seek(0)
    return buffer.getvalue()
