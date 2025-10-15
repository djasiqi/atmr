import logging
import os
import tempfile
from io import BytesIO

from qrbill import QRBill
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

from models import CompanyBillingSettings

app_logger = logging.getLogger("qrbill_service")


class QRBillService:
    """Service pour la génération de QR-Bill"""

    def __init__(self):
        pass

    def generate_qr_bill_svg(self, invoice):
        """Génère un QR-Bill SVG pour une facture"""
        try:
            # Récupérer les paramètres de facturation
            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=invoice.company_id
            ).first()

            if not billing_settings or not billing_settings.iban:
                app_logger.warning(f"Pas d'IBAN configuré pour l'entreprise {invoice.company_id}")
                return None

            # Récupérer les informations de la facture
            company = invoice.company
            client = invoice.client

            # Débiteur : Institution (si facturation tierce) ou Client (si facturation directe)
            if invoice.bill_to_client_id and invoice.bill_to_client_id != invoice.client_id:
                # 🏥 Facturation tierce : débiteur = institution payeuse
                from models import Client as ClientModel
                institution = ClientModel.query.get(invoice.bill_to_client_id)

                if institution and institution.is_institution:
                    debtor_name = institution.institution_name or 'Institution'
                    debtor_street = institution.billing_address or institution.contact_address or 'Adresse non renseignée'
                    # Extraire code postal et ville de l'adresse si possible
                    debtor_pcode = '1200'
                    debtor_city = 'Genève'
                else:
                    debtor_name = 'Institution'
                    debtor_street = 'Adresse non renseignée'
                    debtor_pcode = '1200'
                    debtor_city = 'Genève'
            else:
                # 👤 Facturation directe : débiteur = client (avec même logique que le PDF)
                debtor_name = f"{client.user.first_name or ''} {client.user.last_name or ''}".strip() or client.user.username or 'Client'

                # Récupérer l'adresse avec priorités multiples
                debtor_street = 'Adresse non renseignée'
                debtor_pcode = '1200'
                debtor_city = 'Genève'

                # Priorité 1: Adresse du domicile
                if hasattr(client, 'domicile_address') and client.domicile_address:
                    debtor_street = client.domicile_address
                    if hasattr(client, 'domicile_zip') and client.domicile_zip:
                        debtor_pcode = client.domicile_zip
                    if hasattr(client, 'domicile_city') and client.domicile_city:
                        debtor_city = client.domicile_city
                # Priorité 2: Adresse de l'utilisateur
                elif hasattr(client, 'user') and client.user and hasattr(client.user, 'address') and client.user.address:
                    full_address = client.user.address
                    # Format: "Allée de la Pépinière, 41, 74160, Archamps, France"
                    parts = [p.strip() for p in full_address.split(',')]
                    if len(parts) >= 2:
                        # Rue + numéro
                        debtor_street = f"{parts[0]}, {parts[1]}"
                    if len(parts) >= 3:
                        # Code postal
                        debtor_pcode = parts[2]
                    if len(parts) >= 4:
                        # Ville
                        debtor_city = parts[3]

            # Créer le QR-Bill avec la vraie bibliothèque qrbill
            qr_bill = QRBill(
                account=billing_settings.iban,
                creditor={
                    'name': company.name or 'Emmenez Moi',
                    'street': company.address or 'Route de Chevrens 145',
                    'pcode': '1247',
                    'city': 'Anières',
                    'country': 'CH'
                },
                debtor={
                    'name': debtor_name,
                    'street': debtor_street,
                    'pcode': debtor_pcode,
                    'city': debtor_city,
                    'country': 'CH'
                },
                amount=str(invoice.total_amount),
                currency='CHF',
                reference_number=None,  # Pas de référence QR pour l'instant
                additional_information=f"Facture {invoice.invoice_number} - Période: {invoice.period_month:02d}.{invoice.period_year}",
                language='de'
            )

            # Générer le SVG du QR-Bill
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.svg', delete=False) as temp_svg:
                qr_bill.as_svg(temp_svg.name)

                # Lire le contenu SVG
                with open(temp_svg.name, encoding='utf-8') as f:
                    svg_content = f.read()

                # Nettoyer le fichier temporaire
                os.unlink(temp_svg.name)

                app_logger.info(f"QR-Bill SVG généré pour facture {invoice.invoice_number}")
                return svg_content.encode('utf-8')

        except Exception as e:
            app_logger.error(f"Erreur lors de la génération du QR-Bill SVG: {str(e)}")
            return None

    def generate_qr_bill(self, invoice):
        """Génère un QR-Bill pour une facture"""
        try:
            # Récupérer les paramètres de facturation
            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=invoice.company_id
            ).first()

            if not billing_settings or not billing_settings.iban:
                app_logger.warning(f"Pas d'IBAN configuré pour l'entreprise {invoice.company_id}")
                return None

            # Récupérer les informations de la facture
            company = invoice.company
            client = invoice.client

            # Débiteur : Institution (si facturation tierce) ou Client (si facturation directe)
            if invoice.bill_to_client_id and invoice.bill_to_client_id != invoice.client_id:
                # 🏥 Facturation tierce : débiteur = institution payeuse
                from models import Client as ClientModel
                institution = ClientModel.query.get(invoice.bill_to_client_id)

                if institution and institution.is_institution:
                    debtor_name = institution.institution_name or 'Institution'
                    debtor_street = institution.billing_address or institution.contact_address or 'Adresse non renseignée'
                    # Extraire code postal et ville de l'adresse si possible
                    debtor_pcode = '1200'
                    debtor_city = 'Genève'
                else:
                    debtor_name = 'Institution'
                    debtor_street = 'Adresse non renseignée'
                    debtor_pcode = '1200'
                    debtor_city = 'Genève'
            else:
                # 👤 Facturation directe : débiteur = client (avec même logique que le PDF)
                debtor_name = f"{client.user.first_name or ''} {client.user.last_name or ''}".strip() or client.user.username or 'Client'

                # Récupérer l'adresse avec priorités multiples
                debtor_street = 'Adresse non renseignée'
                debtor_pcode = '1200'
                debtor_city = 'Genève'

                # Priorité 1: Adresse du domicile
                if hasattr(client, 'domicile_address') and client.domicile_address:
                    debtor_street = client.domicile_address
                    if hasattr(client, 'domicile_zip') and client.domicile_zip:
                        debtor_pcode = client.domicile_zip
                    if hasattr(client, 'domicile_city') and client.domicile_city:
                        debtor_city = client.domicile_city
                # Priorité 2: Adresse de l'utilisateur
                elif hasattr(client, 'user') and client.user and hasattr(client.user, 'address') and client.user.address:
                    full_address = client.user.address
                    # Format: "Allée de la Pépinière, 41, 74160, Archamps, France"
                    parts = [p.strip() for p in full_address.split(',')]
                    if len(parts) >= 2:
                        # Rue + numéro
                        debtor_street = f"{parts[0]}, {parts[1]}"
                    if len(parts) >= 3:
                        # Code postal
                        debtor_pcode = parts[2]
                    if len(parts) >= 4:
                        # Ville
                        debtor_city = parts[3]

            # Créer le QR-Bill avec la vraie bibliothèque qrbill
            qr_bill = QRBill(
                account=billing_settings.iban,
                creditor={
                    'name': company.name or 'Emmenez Moi',
                    'street': company.address or 'Route de Chevrens 145',
                    'pcode': '1247',
                    'city': 'Anières',
                    'country': 'CH'
                },
                debtor={
                    'name': debtor_name,
                    'street': debtor_street,
                    'pcode': debtor_pcode,
                    'city': debtor_city,
                    'country': 'CH'
                },
                amount=str(invoice.total_amount),
                currency='CHF',
                reference_number=None,  # Pas de référence QR pour l'instant
                additional_information=f"Facture {invoice.invoice_number} - Période: {invoice.period_month:02d}.{invoice.period_year}",
                language='de'
            )

            # Générer le PDF du QR-Bill
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.svg', delete=False) as temp_svg:
                qr_bill.as_svg(temp_svg.name)

                # Convertir SVG en PDF
                drawing = svg2rlg(temp_svg.name)

                # Créer le PDF en mémoire
                pdf_buffer = BytesIO()
                renderPDF.drawToFile(drawing, pdf_buffer)
                pdf_buffer.seek(0)

                # Nettoyer le fichier temporaire
                os.unlink(temp_svg.name)

                app_logger.info(f"QR-Bill généré pour facture {invoice.invoice_number}")
                return pdf_buffer.getvalue()

        except Exception as e:
            app_logger.error(f"Erreur lors de la génération du QR-Bill: {str(e)}")
            return None

    def generate_qr_reference(self, invoice):
        """Génère une référence QR pour une facture"""
        try:
            # Générer une référence QR basée sur l'ID de la facture
            # Format: 27 caractères (modulo 10) - doit commencer par "RF"
            invoice_id_str = str(invoice.id).zfill(7)
            qr_reference = f"RF{invoice_id_str}"

            # Calculer le check digit (modulo 10)
            check_digit = self._calculate_check_digit(qr_reference)
            qr_reference += str(check_digit)

            # S'assurer que la référence fait exactement 27 caractères
            while len(qr_reference) < 27:
                qr_reference += "0"

            return qr_reference[:27]  # Limiter à 27 caractères

        except Exception as e:
            app_logger.error(f"Erreur lors de la génération de la référence QR: {str(e)}")
            return None

    def _calculate_check_digit(self, reference):
        """Calcule le check digit pour une référence QR"""
        # Algorithme modulo 10 pour les références QR
        weights = [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]

        total = 0
        for i, char in enumerate(reference):
            if char.isdigit():
                total += int(char) * weights[i % len(weights)]
            else:
                # Pour les lettres, utiliser leur valeur ASCII
                total += (ord(char) - ord('A') + 10) * weights[i % len(weights)]

        remainder = total % 10
        return (10 - remainder) % 10
