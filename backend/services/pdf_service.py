import logging
from datetime import UTC, datetime

# Constantes pour éviter les valeurs magiques
from pathlib import Path

from flask import current_app
from sqlalchemy.orm import joinedload

from models import Client, CompanyBillingSettings, Invoice, InvoiceLineType
from services.qrbill_service import QRBillService

LEVEL_ONE = 1
LEVEL_THRESHOLD = 2
MAX_PATIENT_NAME_LENGTH = 18

app_logger = logging.getLogger("pdf_service")


class PDFService:
    """Service pour la génération de PDF de factures et rappels."""

    def __init__(self):
        super().__init__()
        self.qrbill_service = QRBillService()
        self.uploads_dir = Path(Path(Path(__file__).parent.parent), "uploads")
        self.invoices_dir = Path(self.uploads_dir, "invoices")

        # Créer les dossiers s'ils n'existent pas
        Path(self.invoices_dir, exist_ok=True).mkdir(parents=True, exist_ok=True)

    def generate_invoice_pdf(self, invoice):
        """Génère le PDF d'une facture."""
        try:
            # Charger la facture avec toutes les relations
            invoice = Invoice.query.options(
                joinedload(Invoice.company),
                joinedload(Invoice.client).joinedload(Client.user),
                joinedload(Invoice.lines),
                joinedload(Invoice.payments)
            ).filter_by(id=invoice.id).first()

            if not invoice:
                msg = "Facture non trouvée"
                raise ValueError(msg)

            # Générer le contenu PDF
            pdf_content = self._create_invoice_pdf_content(invoice)

            # Sauvegarder le fichier
            filename = f"invoice_{invoice.invoice_number}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = Path(self.invoices_dir, filename)

            with filepath.open("wb") as f:
                f.write(pdf_content)

            # ✅ URL dynamique depuis config
            pdf_base_url = current_app.config.get("PDF_BASE_URL", "http://localhost:5000")
            uploads_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")

            pdf_url = f"{pdf_base_url}{uploads_base}/invoices/{filename}"

            app_logger.info("PDF de facture généré: %s", pdf_url)
            return pdf_url

        except Exception as e:
            app_logger.error("Erreur lors de la génération du PDF de facture: %s", str(e))
            raise

    def generate_reminder_pdf(self, invoice, level):
        """Génère le PDF d'un rappel."""
        try:
            # Charger la facture avec toutes les relations
            invoice = Invoice.query.options(
                joinedload(Invoice.company),
                joinedload(Invoice.client).joinedload(Client.user),
                joinedload(Invoice.lines),
                joinedload(Invoice.payments),
                joinedload(Invoice.reminders)
            ).filter_by(id=invoice.id).first()

            if not invoice:
                msg = "Facture non trouvée"
                raise ValueError(msg)

            # Générer le contenu PDF du rappel
            pdf_content = self._create_reminder_pdf_content(invoice, level)

            # Sauvegarder le fichier
            filename = f"reminder_{invoice.invoice_number}_level{level}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = Path(self.invoices_dir, filename)

            with filepath.open("wb") as f:
                f.write(pdf_content)

            # ✅ URL dynamique
            pdf_base_url = current_app.config.get("PDF_BASE_URL", "http://localhost:5000")
            uploads_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")
            pdf_url = f"{pdf_base_url}{uploads_base}/invoices/{filename}"

            app_logger.info("PDF de rappel généré: %s", pdf_url)
            return pdf_url

        except Exception as e:
            app_logger.error("Erreur lors de la génération du PDF de rappel: %s", str(e))
            raise

    def _create_invoice_pdf_content(self, invoice):
        """Crée le contenu PDF d'une facture en reproduisant exactement le design de référence."""
        # Import ici pour éviter les problèmes de dépendances circulaires
        from io import BytesIO

        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                              topMargin=2*cm, bottomMargin=2*cm,
                              leftMargin=2*cm, rightMargin=2*cm)

        # Styles basés sur le design de référence
        styles = getSampleStyleSheet()


        # Style pour le texte normal
        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceAfter=6,
            fontName="Helvetica"
        )



        # Contenu
        story = []

        # === EN-TÊTE AVEC LOGO ET INFORMATIONS ENTREPRISE ===
        company = invoice.company

        # Logo de l'entreprise
        logo_img = None
        logo_path = None
        logo_width = 0.0
        logo_height = 0.0
        if hasattr(company, "logo_url") and company.logo_url:
            try:
                # Corriger le chemin du logo (éviter le double /uploads)
                logo_url_clean = company.logo_url.lstrip("/")
                if logo_url_clean.startswith("uploads/"):
                    logo_url_clean = logo_url_clean[8:]  # Supprimer 'uploads/'

                logo_path = Path(Path(Path(__file__).parent.parent), "uploads", logo_url_clean)

                if Path(logo_path).exists():
                    # Calculer les proportions du logo original (2251x540)
                    # Ratio largeur/hauteur = 2251/540 ≈ 4.17
                    # Utiliser des pourcentages pour s'adapter à tous les formats
                    logo_width_percent = 0.15  # 15% de la largeur de page

                    # Convertir en points (1 inch = 72 points, page A4 ≈ 21cm = 595 points)
                    logo_width = 595 * logo_width_percent  # ≈ 89 points
                    logo_height = logo_width / 4.17  # ≈ 21 points

                    # Vérifier si c'est un fichier SVG
                    if logo_path.suffix.lower() == '.svg':
                        # Convertir SVG en drawing ReportLab avec svglib
                        try:
                            from svglib.svglib import svg2rlg
                            drawing = svg2rlg(str(logo_path))
                            if drawing:
                                # Redimensionner le drawing
                                original_width = drawing.width
                                original_height = drawing.height
                                if original_width > 0 and original_height > 0:
                                    scale_x = logo_width / original_width
                                    scale_y = logo_height / original_height
                                    drawing.scale(scale_x, scale_y)
                                logo_img = drawing
                            else:
                                app_logger.warning("Impossible de convertir le SVG en drawing: %s", logo_path)
                        except Exception as svg_error:
                            app_logger.error("Erreur lors de la conversion SVG: %s - %s", logo_path, str(svg_error))
                            # En cas d'erreur, on continue sans logo plutôt que de faire échouer la génération
                            logo_img = None
                    else:
                        # Pour les autres formats (PNG, JPG, etc.), utiliser Image directement
                        logo_img = Image(logo_path, width=logo_width, height=logo_height)
            except Exception as e:
                app_logger.warning("Impossible de charger le logo: %s", e)

        # Informations de l'entreprise
        company_name = company.name or "Emmenez Moi"
        company_address = company.address or "Route de Chevrens 145, 1247 Anières"
        company_phone = company.contact_phone or "0225120203"
        company_email = company.billing_email or company.contact_email or "info@casa-famiglia.ch"
        company_uid = company.uid_ide or "CHE-27348.653"

        # === LOGO ET COORDONNÉES ENTREPRISE (GAUCHE) ===
        if logo_img:
            # Vérifier si c'est un drawing (SVG converti) en vérifiant la présence d'attributs spécifiques
            # Les drawings de svglib ont des attributs width, height et une méthode scale
            is_drawing = hasattr(logo_img, 'width') and hasattr(logo_img, 'height') and hasattr(logo_img, 'scale')
            
            if is_drawing:
                # Pour les drawings SVG, créer un tableau pour l'alignement
                # Table et TableStyle sont déjà importés plus haut
                logo_table = Table([[logo_img]], colWidths=[logo_width])
                logo_table.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(logo_table)
            else:
                # Pour les images standard (PNG, JPG), utiliser un Paragraph
                from reportlab.lib.styles import ParagraphStyle
                logo_style = ParagraphStyle(
                    "LogoStyle",
                    parent=styles["Normal"],
                    alignment=TA_LEFT,
                    leftIndent=0,
                    rightIndent=0,
                    spaceAfter=8
                )
                # Créer un Paragraph avec le logo
                logo_para = Paragraph(f'<img src="{logo_path}" width="{logo_width}" height="{logo_height}"/>', logo_style)
                story.append(logo_para)

        # Coordonnées entreprise alignées à gauche
        company_info_left = f"""
        {company_name}<br/>
        {company_address}<br/>
        Email facturation : {company_email}<br/>
        Téléphone : {company_phone}<br/>
        IDE/UID : {company_uid}
        """

        story.append(Paragraph(company_info_left, normal_style))
        story.append(Spacer(1, 20))

        # === INFORMATIONS CLIENT OU INSTITUTION (DROITE) ===
        # Si facturation tierce, afficher l'institution, sinon le client
        if invoice.bill_to_client_id and invoice.bill_to_client_id != invoice.client_id:
            # 🏥 Facturation tierce : afficher l'institution payeuse
            from models import Client as ClientModel
            institution = ClientModel.query.get(invoice.bill_to_client_id)

            if institution and institution.is_institution:
                billed_to_name = institution.institution_name or "Institution"
                billed_to_address = institution.billing_address or institution.contact_address or "Adresse non renseignée"
            else:
                # Fallback si l'institution n'est pas trouvée
                billed_to_name = "Institution"
                billed_to_address = "Adresse non renseignée"
        else:
            # 👤 Facturation directe : afficher le client
            client = invoice.client
            billed_to_name = f"{client.user.first_name or ''} {client.user.last_name or ''}".strip() or client.user.username or "Client"

            # Adresse du client formatée correctement (sans <br/>)
            billed_to_address = "Adresse non renseignée"
            if hasattr(client, "domicile_address") and client.domicile_address:
                street_address = client.domicile_address
                if hasattr(client, "domicile_zip") and hasattr(client, "domicile_city") and client.domicile_zip and client.domicile_city:
                    billed_to_address = f"{street_address}\n{client.domicile_zip} {client.domicile_city} Suisse"
                else:
                    billed_to_address = street_address
            elif hasattr(client, "user") and client.user and hasattr(client.user, "address") and client.user.address:
                billed_to_address = client.user.address

        # Informations de facturation alignées à droite
        billed_to_info_right = f"""
        <para align="right">
        <b>Facturé à :</b><br/>
        {billed_to_name}<br/>
        {billed_to_address}
        </para>
        """

        story.append(Paragraph(billed_to_info_right, normal_style))
        story.append(Spacer(1, 20))

        # === INFORMATIONS FACTURE (GAUCHE) ===
        invoice_info_left = f"""
        <b>Numéro de facture :</b> {invoice.invoice_number}<br/>
        <b>Date d'émission :</b> {invoice.issued_at.strftime('%d.%m.%Y')}<br/>
        <b>Date d'échéance :</b> {invoice.due_date.strftime('%d.%m.%Y')}<br/>
        <b>Période :</b> {invoice.period_month:02d}.{invoice.period_year}
        """

        story.append(Paragraph(invoice_info_left, normal_style))
        story.append(Spacer(1, 20))

        # === TABLEAU DES COURSES ===
        # Fonction pour formater les adresses longues avec retour à la ligne dans les colonnes
        def format_address_for_table(address, max_length=25):
            if not address or address == "Adresse inconnue":
                return "Adresse non renseignée"

            # Nettoyer l'adresse : supprimer "Suisse" et "Trajet" mais garder les numéros d'adresse
            clean_address = address.replace(", Suisse", "").strip()
            # Supprimer le mot "Trajet" au début
            import re
            clean_address = re.sub(r"^Trajet\s+", "", clean_address)
            # Supprimer "Suisse" à la fin
            clean_address = clean_address.replace(" Suisse", "").strip()
            # Supprimer les points médians (·) mais garder les numéros d'adresse
            clean_address = clean_address.replace(" · ", " ").replace("·", "")

            # Si l'adresse est courte, la retourner telle quelle
            if len(clean_address) <= max_length:
                return clean_address

            # Diviser l'adresse en mots et créer des lignes
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

            return "\n".join(lines[:3])  # Maximum 3 lignes avec \n au lieu de <br/>

        # Vérifier si c'est une facturation tierce pour ajouter la colonne "Patient"
        is_third_party = invoice.bill_to_client_id and invoice.bill_to_client_id != invoice.client_id

        # En-têtes du tableau
        if is_third_party:
            table_data = [["Date", "Patient", "Départ", "Arrivée", "Montant"]]
        else:
            table_data = [["Date", "Départ", "Arrivée", "Montant"]]

        # Ajouter les lignes de facture - utiliser les vraies données des bookings
        for line in invoice.lines:
            # Pour les lignes de réservation, extraire les informations
            if line.type == InvoiceLineType.RIDE and line.reservation_id:
                # Essayer de récupérer les informations de la réservation
                from models import Booking
                booking = Booking.query.get(line.reservation_id)

                if booking:
                    # Récupérer la vraie date de la course
                    date_str = booking.scheduled_time.strftime("%d/%m/%Y") if booking.scheduled_time else ""

                    # Nom du patient (pour facturation tierce)
                    patient_name = ""
                    if is_third_party:
                        patient_name = booking.customer_name or f"{booking.client.user.first_name or ''} {booking.client.user.last_name or ''}".strip() or "Patient"
                        # Tronquer si trop long
                        if len(patient_name) > MAX_PATIENT_NAME_LENGTH:
                            patient_name = patient_name[:MAX_PATIENT_NAME_LENGTH - 1] + "."

                    # Supprimer le mot "Trajet" et nettoyer les adresses
                    departure = format_address_for_table(booking.pickup_location, max_length=20 if is_third_party else 25)
                    arrival = format_address_for_table(booking.dropoff_location, max_length=20 if is_third_party else 25)
                    amount = f"{line.line_total:.2f}"

                    # Construire la ligne selon le type de facturation
                    if is_third_party:
                        table_data.append([date_str, patient_name, departure, arrival, amount])
                    else:
                        table_data.append([date_str, departure, arrival, amount])
                else:
                    # Si pas de booking trouvé, utiliser la description mais séparer départ/arrivée
                    date_str = ""
                    desc = line.description
                    if " → " in desc:
                        parts = desc.split(" → ")
                        departure = format_address_for_table(parts[0], max_length=20 if is_third_party else 25)
                        arrival = format_address_for_table(parts[1], max_length=20 if is_third_party else 25) if len(parts) > 1 else ""
                    else:
                        departure = format_address_for_table(desc, max_length=20 if is_third_party else 25)
                        arrival = ""
                    amount = f"{line.line_total:.2f}"

                    # Construire la ligne selon le type de facturation
                    if is_third_party:
                        table_data.append([date_str, "N/A", departure, arrival, amount])
                    else:
                        table_data.append([date_str, departure, arrival, amount])
            else:
                # Pour les autres types de lignes, utiliser la description
                date_str = ""
                departure = format_address_for_table(line.description, max_length=20 if is_third_party else 25)
                arrival = ""
                amount = f"{line.line_total:.2f}"

                # Construire la ligne selon le type de facturation
                if is_third_party:
                    table_data.append([date_str, "N/A", departure, arrival, amount])
                else:
                    table_data.append([date_str, departure, arrival, amount])

        # Adapter les largeurs de colonnes selon le type de facturation
        if is_third_party:
            services_table = Table(table_data, colWidths=[2*cm, 3*cm, 4.5*cm, 4.5*cm, 2.5*cm])
        else:
            services_table = Table(table_data, colWidths=[2.5*cm, 6*cm, 6*cm, 2.5*cm])
        services_table.setStyle(TableStyle([
            # En-têtes
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("ALIGN", (3, 0), (3, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            # Corps du tableau
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
            ("TOPPADDING", (0, 1), (-1, -1), 8),
            # Lignes de séparation fines entre les lignes
            ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.lightgrey),
        ]))

        story.append(services_table)
        story.append(Spacer(1, 15))

        # === TOTAL ===
        total_amount = float(invoice.total_amount)

        # Ligne de séparation plus épaisse avant le total
        total_separator = Table([[""]], colWidths=[16*cm])
        total_separator.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (0, 0), 1, colors.black),
        ]))
        story.append(total_separator)
        story.append(Spacer(1, 8))

        # Adapter le tableau du total selon le type de facturation
        if is_third_party:
            total_data = [
                ["", "", "", "TOTAL :", f"{total_amount:.2f}"],
            ]
            total_table = Table(total_data, colWidths=[2*cm, 3*cm, 4.5*cm, 2*cm, 2.5*cm])
        else:
            total_data = [
                ["", "", "TOTAL :", f"{total_amount:.2f}"],
            ]
            total_table = Table(total_data, colWidths=[2.5*cm, 6*cm, 2.5*cm, 2.5*cm])
        # Style du tableau du total
        if is_third_party:
            total_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ALIGN", (3, 0), (4, 0), "RIGHT"),  # TOTAL et montant alignés à droite
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]))
        else:
            total_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ALIGN", (2, 0), (3, 0), "RIGHT"),  # TOTAL et montant alignés à droite
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]))

        story.append(total_table)
        story.append(Spacer(1, 30))

        # === PIED DE PAGE - NOTES DE FACTURATION ===

        # Message de facturation (texte exact demandé)
        message = "En votre aimable règlement net sous 10 jours avec nos remerciements anticipés. En cas de retard de paiement, des frais de rappel d'un montant de CHF 15.- vous seront facturés, conformément à nos conditions générales."

        story.append(Paragraph(message, normal_style))
        story.append(Spacer(1, 8))

        # Informations bancaires (texte exact demandé)
        banking_info = "Paiement par virement bancaire : IBAN : CH6509000000152631289"
        story.append(Paragraph(banking_info, normal_style))

        # === QR-BILL SUISSE OFFICIEL SUR PAGE SÉPARÉE ===
        # Forcer une nouvelle page pour le QR-Bill (toujours après la facture)
        from reportlab.platypus import PageBreak
        story.append(PageBreak())

        # Ajouter un espacement pour pousser le QR-Bill vraiment en bas de la page QR-Bill
        story.append(Spacer(1, 545))  # Espacement optimal pour pousser vraiment en bas

        try:
            # Générer le QR-Bill suisse officiel avec la vraie bibliothèque
            qr_bill_service = self.qrbill_service
            qr_bill_svg_content = qr_bill_service.generate_qr_bill_svg(invoice)

            if qr_bill_svg_content:
                # Convertir le SVG directement en drawing ReportLab

                from svglib.svglib import svg2rlg

                # Convertir SVG en drawing ReportLab
                drawing = svg2rlg(BytesIO(qr_bill_svg_content))

                if drawing:
                    # Redimensionner le drawing pour qu'il s'adapte à la page
                    drawing.width = 12*cm
                    drawing.height = 6*cm
                    drawing.scale(12*cm/drawing.width, 6*cm/drawing.height)

                    # Centrer le QR-Bill avec un tableau
                    from reportlab.platypus import Table, TableStyle

                    # Créer un tableau avec colonne vide pour vraiment aligner à gauche
                    qr_table = Table([[drawing, ""]], colWidths=[6*cm, 12*cm])  # QR-Bill encore plus petit + colonne vide encore plus grande
                    qr_table.setStyle(TableStyle([
                        ("ALIGN", (0, 0), (0, 0), "LEFT"),  # QR-Bill à gauche
                        ("ALIGN", (1, 0), (1, 0), "LEFT"),   # Colonne vide à droite
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ]))

                    # Ajouter le QR-Bill centré au PDF sur la deuxième page
                    story.append(qr_table)
            else:
                story.append(Paragraph("QR-Bill non disponible - IBAN non configuré", normal_style))

        except Exception as e:
            app_logger.warning("Impossible de générer le QR-Bill: %s", e)
            story.append(Paragraph("QR-Bill non disponible", normal_style))

        # Générer le PDF
        doc.build(story)

        # Retourner le contenu
        buffer.seek(0)
        return buffer.getvalue()

    def _create_swiss_qr_bill_layout(self, invoice, billing_settings, qr_image):
        """Crée le layout authentique du QR-Bill suisse."""
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, Spacer

        styles = getSampleStyleSheet()

        # Style pour le texte normal
        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceAfter=2,
            fontName="Helvetica"
        )

        # Style pour les titres de section
        section_title_style = ParagraphStyle(
            "SectionTitle",
            parent=styles["Normal"],
            fontSize=12,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT,
            spaceAfter=8,
            textColor=colors.black
        )

        # Style pour les labels
        label_style = ParagraphStyle(
            "Label",
            parent=styles["Normal"],
            fontSize=8,
            fontName="Helvetica",
            alignment=TA_LEFT,
            spaceAfter=2,
            textColor=colors.black
        )

        # Style pour les valeurs
        value_style = ParagraphStyle(
            "Value",
            parent=styles["Normal"],
            fontSize=8,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT,
            spaceAfter=4,
            textColor=colors.black
        )

        # === SECTION GAUCHE: EMPFANGSSCHEIN (Reçu) ===
        left_section = []

        # Titre
        left_section.append(Paragraph("Empfangsschein", section_title_style))

        # Informations créancier
        left_section.append(Paragraph("Konto / Zahlbar an", label_style))
        left_section.append(Paragraph(billing_settings.iban, value_style))
        left_section.append(Paragraph(invoice.company.name or "Emmenez Moi", normal_style))
        left_section.append(Paragraph(invoice.company.address or "Route de Chevrens 145", normal_style))
        left_section.append(Paragraph("1247 Anières", normal_style))
        left_section.append(Spacer(1, 8))

        # Informations débiteur
        left_section.append(Paragraph("Zahlbar durch", label_style))
        left_section.append(Paragraph(f"{invoice.client.user.first_name or ''} {invoice.client.user.last_name or ''}", normal_style))
        left_section.append(Paragraph(invoice.client.domicile_address or "Adresse non renseignée", normal_style))
        left_section.append(Paragraph(f"{invoice.client.domicile_zip or ''} {invoice.client.domicile_city or ''}", normal_style))
        left_section.append(Spacer(1, 8))

        # Référence
        left_section.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        left_section.append(Paragraph(qr_ref, value_style))
        left_section.append(Spacer(1, 8))

        # Montant
        left_section.append(Paragraph("Währung", label_style))
        left_section.append(Paragraph("CHF", value_style))
        left_section.append(Paragraph("Betrag", label_style))
        left_section.append(Paragraph(f"{invoice.total_amount:.2f}", value_style))
        left_section.append(Spacer(1, 20))

        # Annahmestelle
        left_section.append(Paragraph("Annahmestelle", ParagraphStyle("Center", parent=styles["Normal"], fontSize=8, alignment=TA_CENTER)))

        # === SECTION DROITE: ZAHLTEIL (Partie paiement) ===
        right_section = []

        # Titre
        right_section.append(Paragraph("Zahlteil", section_title_style))

        # Informations créancier
        right_section.append(Paragraph("Konto / Zahlbar an", label_style))
        right_section.append(Paragraph(billing_settings.iban, value_style))
        right_section.append(Paragraph(invoice.company.name or "Emmenez Moi", normal_style))
        right_section.append(Paragraph(invoice.company.address or "Route de Chevrens 145", normal_style))
        right_section.append(Paragraph("1247 Anières", normal_style))
        right_section.append(Spacer(1, 8))

        # QR Code
        right_section.append(qr_image)
        right_section.append(Spacer(1, 8))

        # Informations débiteur
        right_section.append(Paragraph("Zahlbar durch", label_style))
        right_section.append(Paragraph(f"{invoice.client.user.first_name or ''} {invoice.client.user.last_name or ''}", normal_style))
        right_section.append(Paragraph(invoice.client.domicile_address or "Adresse non renseignée", normal_style))
        right_section.append(Paragraph(f"{invoice.client.domicile_zip or ''} {invoice.client.domicile_city or ''}", normal_style))
        right_section.append(Spacer(1, 8))

        # Référence
        right_section.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        right_section.append(Paragraph(qr_ref, value_style))
        right_section.append(Spacer(1, 8))

        # Montant
        right_section.append(Paragraph("Währung", label_style))
        right_section.append(Paragraph("CHF", value_style))
        right_section.append(Paragraph("Betrag", label_style))
        right_section.append(Paragraph(f"{invoice.total_amount:.2f}", value_style))

        # === LIGNE DE COUPE ===
        cut_line = [Paragraph("✂", ParagraphStyle("CutLine", parent=styles["Normal"], fontSize=12, alignment=TA_CENTER))]

        # Retourner les données du tableau
        return [
            [left_section, cut_line, right_section]
        ]

    def _create_official_swiss_qr_bill(self, invoice, billing_settings, qr_image):
        """Crée un QR-Bill suisse officiel avec le format exact."""
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

        styles = getSampleStyleSheet()

        # Styles spécifiques pour le QR-Bill suisse
        title_style = ParagraphStyle(
            "QRTitle",
            parent=styles["Normal"],
            fontSize=11,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT,
            spaceAfter=6,
            textColor=colors.black
        )

        label_style = ParagraphStyle(
            "QRLabel",
            parent=styles["Normal"],
            fontSize=7,
            fontName="Helvetica",
            alignment=TA_LEFT,
            spaceAfter=1,
            textColor=colors.black
        )

        value_style = ParagraphStyle(
            "QRValue",
            parent=styles["Normal"],
            fontSize=7,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT,
            spaceAfter=3,
            textColor=colors.black
        )

        normal_text_style = ParagraphStyle(
            "QRNormal",
            parent=styles["Normal"],
            fontSize=7,
            fontName="Helvetica",
            alignment=TA_LEFT,
            spaceAfter=1,
            textColor=colors.black
        )

        # === CONSTRUCTION DU QR-BILL ===

        # Section gauche - Empfangsschein
        left_content = []
        left_content.append(Paragraph("Empfangsschein", title_style))
        left_content.append(Spacer(1, 4))

        # Konto / Zahlbar an
        left_content.append(Paragraph("Konto / Zahlbar an", label_style))
        left_content.append(Paragraph(billing_settings.iban, value_style))
        left_content.append(Paragraph(invoice.company.name or "Emmenez Moi", normal_text_style))
        left_content.append(Paragraph(invoice.company.address or "Route de Chevrens 145", normal_text_style))
        left_content.append(Paragraph("1247 Anières", normal_text_style))
        left_content.append(Spacer(1, 6))

        # Zahlbar durch
        left_content.append(Paragraph("Zahlbar durch", label_style))
        left_content.append(Paragraph(f"{invoice.client.user.first_name or ''} {invoice.client.user.last_name or ''}", normal_text_style))
        left_content.append(Paragraph(invoice.client.domicile_address or "Adresse non renseignée", normal_text_style))
        left_content.append(Paragraph(f"{invoice.client.domicile_zip or ''} {invoice.client.domicile_city or ''}", normal_text_style))
        left_content.append(Spacer(1, 6))

        # Referenz
        left_content.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        left_content.append(Paragraph(qr_ref, value_style))
        left_content.append(Spacer(1, 6))

        # Währung et Betrag
        left_content.append(Paragraph("Währung", label_style))
        left_content.append(Paragraph("CHF", value_style))
        left_content.append(Paragraph("Betrag", label_style))
        left_content.append(Paragraph(f"{invoice.total_amount:.2f}", value_style))
        left_content.append(Spacer(1, 20))

        # Annahmestelle
        left_content.append(Paragraph("Annahmestelle", ParagraphStyle("Center", parent=styles["Normal"], fontSize=7, alignment=TA_CENTER)))

        # Section droite - Zahlteil
        right_content = []
        right_content.append(Paragraph("Zahlteil", title_style))
        right_content.append(Spacer(1, 4))

        # Konto / Zahlbar an
        right_content.append(Paragraph("Konto / Zahlbar an", label_style))
        right_content.append(Paragraph(billing_settings.iban, value_style))
        right_content.append(Paragraph(invoice.company.name or "Emmenez Moi", normal_text_style))
        right_content.append(Paragraph(invoice.company.address or "Route de Chevrens 145", normal_text_style))
        right_content.append(Paragraph("1247 Anières", normal_text_style))
        right_content.append(Spacer(1, 6))

        # QR Code
        right_content.append(qr_image)
        right_content.append(Spacer(1, 6))

        # Zahlbar durch
        right_content.append(Paragraph("Zahlbar durch", label_style))
        right_content.append(Paragraph(f"{invoice.client.user.first_name or ''} {invoice.client.user.last_name or ''}", normal_text_style))
        right_content.append(Paragraph(invoice.client.domicile_address or "Adresse non renseignée", normal_text_style))
        right_content.append(Paragraph(f"{invoice.client.domicile_zip or ''} {invoice.client.domicile_city or ''}", normal_text_style))
        right_content.append(Spacer(1, 6))

        # Referenz
        right_content.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        right_content.append(Paragraph(qr_ref, value_style))
        right_content.append(Spacer(1, 6))

        # Währung et Betrag
        right_content.append(Paragraph("Währung", label_style))
        right_content.append(Paragraph("CHF", value_style))
        right_content.append(Paragraph("Betrag", label_style))
        right_content.append(Paragraph(f"{invoice.total_amount:.2f}", value_style))

        # Créer le tableau avec ligne de coupe
        qr_bill_data = [
            [left_content, "", right_content]
        ]

        # Tableau QR-Bill avec ligne de coupe
        qr_bill_table = Table(qr_bill_data, colWidths=[8.5*cm, 0.3*cm, 8.5*cm])
        qr_bill_table.setStyle(TableStyle([
            # Bordures extérieures
            ("BOX", (0, 0), (-1, -1), 1, colors.black),

            # Ligne de coupe verticale
            ("LINEBEFORE", (1, 0), (1, -1), 1, colors.black),

            # Alignement
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            # Padding
            ("PADDING", (0, 0), (0, -1), 8),  # Section gauche
            ("PADDING", (2, 0), (2, -1), 8),  # Section droite
            ("PADDING", (1, 0), (1, -1), 0),  # Ligne de coupe

            # Fond blanc
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ]))

        return qr_bill_table

    def _create_reminder_pdf_content(self, invoice, level):
        """Crée le contenu PDF d'un rappel."""
        # Import ici pour éviter les problèmes de dépendances circulaires
        from io import BytesIO

        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.red,
            alignment=TA_CENTER,
            spaceAfter=30
        )

        # Contenu
        story = []

        # Titre selon le niveau
        if level == LEVEL_ONE:
            title = "RAPPEL DE PAIEMENT"
        elif level == LEVEL_THRESHOLD:
            title = "DEUXIÈME RAPPEL DE PAIEMENT"
        else:
            title = "DERNIER RAPPEL DE PAIEMENT"

        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))

        # Informations de la facture
        invoice_info = [
            ["Numéro de facture:", invoice.invoice_number],
            ["Date d'émission:", invoice.issued_at.strftime("%d.%m.%Y")],
            ["Date d'échéance:", invoice.due_date.strftime("%d.%m.%Y")],
            ["Montant dû:", f"{invoice.balance_due:.2f}"],
        ]

        invoice_table = Table(invoice_info, colWidths=[6*cm, 6*cm])
        invoice_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]))

        story.append(invoice_table)
        story.append(Spacer(1, 30))

        # Informations du client
        client = invoice.client
        client_name = f"{client.user.first_name or ''} {client.user.last_name or ''}".strip() or client.user.username or "Client"

        story.append(Paragraph(f"Cher/Chère {client_name},", styles["Normal"]))
        story.append(Spacer(1, 20))

        # Message selon le niveau
        billing_settings = CompanyBillingSettings.query.filter_by(company_id=invoice.company_id).first()

        if level == LEVEL_ONE:
            if billing_settings and billing_settings.reminder1template:
                message = billing_settings.reminder1template
            else:
                message = f"Nous vous rappelons que votre facture {invoice.invoice_number} d'un montant de {invoice.balance_due:.2f}"
        elif level == LEVEL_THRESHOLD:
            if billing_settings and billing_settings.reminder2template:
                message = billing_settings.reminder2template
            else:
                message = f"Conformément à nos CG, un montant de CHF 40.- a été ajouté à votre facture {invoice.invoice_number}. À défaut de règlement dans ce délai, une procédure de mise en demeure sera engagée."
        elif billing_settings and billing_settings.reminder3template:
            message = billing_settings.reminder3template
        else:
            message = "Dernier rappel : Merci d'effectuer votre règlement net sous 5 jours. En l'absence de paiement, une mise en demeure sera engagée, entraînant des frais supplémentaires et une éventuelle procédure légale."

        story.append(Paragraph(message, styles["Normal"]))
        story.append(Spacer(1, 20))

        # Informations bancaires
        if billing_settings and billing_settings.iban:
            banking_info = f"Paiement par virement bancaire : IBAN : {billing_settings.iban}"
            story.append(Paragraph(banking_info, styles["Normal"]))

        # Générer le PDF
        doc.build(story)

        # Retourner le contenu
        buffer.seek(0)
        return buffer.getvalue()
