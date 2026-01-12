# ruff: noqa: G004
import logging
import tempfile
from io import BytesIO
from pathlib import Path

from qrbill import QRBill  # pyright: ignore[reportMissingModuleSource]
from reportlab.graphics import renderPDF  # pyright: ignore[reportMissingModuleSource]
from svglib.svglib import svg2rlg  # pyright: ignore[reportMissingImports]

from models import CompanyBillingSettings
from services.billing import BillingProfileService, generate_scor_reference

# Constantes pour éviter les valeurs magiques
MIN_ADDRESS_PARTS = 2
MIN_ADDRESS_PARTS_POSTAL = 3
MIN_ADDRESS_PARTS_CITY = 4
QR_REFERENCE_LENGTH = 27

app_logger = logging.getLogger("qrbill_service")


class QRBillService:
    """Service pour la génération de QR-Bill."""

    def __init__(self):
        super().__init__()

    def _get_payment_reference(self, invoice):
        """Génère la référence de paiement selon le mode configuré.

        Args:
            invoice: Facture pour laquelle générer la référence

        Returns:
            str | None: Référence de paiement (SCOR/QRR) ou None
        """
        try:
            # Récupérer le profil de facturation
            profile = BillingProfileService.get_by_company_id(invoice.company_id)

            if not profile:
                app_logger.warning(
                    "[QR-Bill] Pas de profil pour company_id=%s, pas de référence générée",
                    invoice.company_id,
                )
                return None

            # Vérifier le mode de référence
            if profile.payment_reference_mode == "NONE":
                app_logger.debug("[QR-Bill] Mode NONE : pas de référence")
                return None

            if profile.payment_reference_mode == "SCOR":
                # Générer une référence SCOR (ISO 11649)
                app_logger.debug(
                    "[QR-Bill] Génération SCOR pour %s", invoice.invoice_number
                )
                return generate_scor_reference(
                    invoice.invoice_number, company_id=invoice.company_id
                )

            if profile.payment_reference_mode == "QRR":
                # TODO: Implémenter QRR (nécessite QR-IBAN)
                app_logger.warning(
                    "[QR-Bill] Mode QRR non encore implémenté, fallback sur SCOR"
                )
                return generate_scor_reference(
                    invoice.invoice_number, company_id=invoice.company_id
                )

            app_logger.error(
                "[QR-Bill] Mode de référence inconnu : %s",
                profile.payment_reference_mode,
            )
            return None

        except Exception as e:
            app_logger.error(f"[QR-Bill] Erreur génération référence : {e}")
            return None

    def _get_creditor_info(self, company):
        """✅ Méthode helper pour obtenir toutes les infos du créancier (entreprise).

        Utilise CompanyBillingProfile comme source unique.
        Retourne adresse + IBAN + mode de paiement.

        Returns:
            dict: {
                'address': {...},  # Adresse structurée
                'iban': str,       # IBAN du profil
                'address_type': 'S' ou 'K'  # Type d'adresse QR-Bill
            }
        """
        # Essayer de récupérer le profil de facturation
        profile = BillingProfileService.get_by_company_id(company.id)

        if profile:
            # ✅ Utiliser le profil comme source unique
            # Type S (Structured) : rue et numéro séparés
            # Si building_number est vide, street_name contient déjà l'adresse complète
            if profile.building_number and profile.building_number.strip():
                creditor_street = (
                    f"{profile.street_name} {profile.building_number}".strip()
                )
            else:
                creditor_street = profile.street_name or ""
            creditor_pcode = profile.postal_code
            creditor_city = profile.city
            creditor_country = profile.country_code
            creditor_name = profile.legal_name

            # IBAN depuis le profil (priorité : qr_iban puis iban)
            iban = profile.qr_iban or profile.iban

            address_type = "S"  # Type structuré

            app_logger.debug(
                "[QR-Bill] Utilisation profil (ID=%s) pour company_id=%s (Type %s)",
                profile.id,
                company.id,
                address_type,
            )
        else:
            # Fallback sur les anciennes données (Type K - Combined)
            app_logger.warning(
                "[QR-Bill] Pas de profil pour company_id=%s, utilisation données company.* (Type K)",
                company.id,
            )
            creditor_street = (
                company.domicile_address_line1
                or company.address
                or "[Adresse non configurée]"
            )
            creditor_pcode = company.domicile_zip or "0000"
            creditor_city = company.domicile_city or "[Ville non configurée]"
            creditor_country = company.domicile_country or "CH"
            creditor_name = company.name or "[Entreprise non configurée]"

            # IBAN depuis company.iban (fallback)
            iban = company.iban or None

            address_type = "K"  # Type combiné (fallback)

        return {
            "address": {
                "name": creditor_name,
                "street": creditor_street,
                "pcode": creditor_pcode,
                "city": creditor_city,
                "country": creditor_country,
            },
            "iban": iban,
            "address_type": address_type,
        }

    def generate_qr_bill_svg(self, invoice):
        """Génère un QR-Bill SVG pour une facture."""
        try:
            # Récupérer les paramètres de facturation
            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=invoice.company_id
            ).first()

            # #region agent log
            import json
            import time
            from pathlib import Path

            log_path = Path("/app/.cursor/debug.log")
            log_data = {
                "location": ("qrbill_service.py:generate_qr_bill_svg:check_iban"),
                "message": "Checking IBAN for QR-Bill generation",
                "data": {
                    "invoice_id": invoice.id,
                    "company_id": invoice.company_id,
                    "billing_settings_found": billing_settings is not None,
                    "has_iban_raw": (
                        hasattr(billing_settings, "_iban_raw")
                        if billing_settings
                        else False
                    ),
                    "iban_raw_value": (
                        str(getattr(billing_settings, "_iban_raw", None))
                        if billing_settings
                        else None
                    ),
                    "iban_decrypted": (
                        billing_settings.iban if billing_settings else None
                    ),
                    "iban_is_none": (
                        billing_settings.iban is None if billing_settings else True
                    ),
                    "iban_is_empty": (
                        billing_settings.iban == "" if billing_settings else True
                    ),
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "I",
            }
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data) + "\n")
            except Exception:
                pass
            # #endregion

            # Récupérer les informations de la facture
            company = invoice.company
            client = invoice.client

            # Débiteur : Institution (si facturation tierce) ou Client
            # (si facturation directe)
            if (
                invoice.bill_to_client_id
                and invoice.bill_to_client_id != invoice.client_id
            ):
                # 🏥 Facturation tierce : débiteur = institution payeuse
                from models import Client as ClientModel

                institution = ClientModel.query.get(invoice.bill_to_client_id)

                if institution and institution.is_institution:
                    debtor_name = institution.institution_name or "Institution"
                    debtor_street = (
                        institution.billing_address
                        or institution.contact_address
                        or "Adresse non renseignée"
                    )
                    # Extraire code postal et ville de l'adresse si possible
                    debtor_pcode = "1200"
                    debtor_city = "Genève"
                else:
                    debtor_name = "Institution"
                    debtor_street = "Adresse non renseignée"
                    debtor_pcode = "1200"
                    debtor_city = "Genève"
            else:
                # 👤 Facturation directe : débiteur = client
                # (avec même logique que le PDF)
                debtor_name = (
                    (
                        f"{client.user.first_name or ''} {client.user.last_name or ''}"
                    ).strip()
                    or client.user.username
                    or "Client"
                )

                # Récupérer l'adresse avec priorités multiples
                debtor_street = "Adresse non renseignée"
                debtor_pcode = "1200"
                debtor_city = "Genève"

                # Priorité 1: Adresse du domicile
                if hasattr(client, "domicile_address") and client.domicile_address:
                    debtor_street = client.domicile_address
                    if hasattr(client, "domicile_zip") and client.domicile_zip:
                        debtor_pcode = client.domicile_zip
                    if hasattr(client, "domicile_city") and client.domicile_city:
                        debtor_city = client.domicile_city
                # Priorité 2: Adresse de l'utilisateur
                elif (
                    hasattr(client, "user")
                    and client.user
                    and hasattr(client.user, "address")
                    and client.user.address
                ):
                    full_address = client.user.address
                    # Format: "Allée de la Pépinière, 41, 74160, Archamps, France"
                    parts = [p.strip() for p in full_address.split(",")]
                    if len(parts) >= MIN_ADDRESS_PARTS:
                        # Rue + numéro
                        debtor_street = f"{parts[0]}, {parts[1]}"
                    if len(parts) >= MIN_ADDRESS_PARTS_POSTAL:
                        # Code postal
                        debtor_pcode = parts[2]
                    if len(parts) >= MIN_ADDRESS_PARTS_CITY:
                        # Ville
                        debtor_city = parts[3]

            # ✅ Utiliser l'adresse de domiciliation (cohérence avec PDF)
            creditor_info = self._get_creditor_info(company)
            creditor_data = creditor_info["address"]
            iban_from_profile = creditor_info["iban"]

            # Utiliser l'IBAN du profil en priorité (fallback sur billing_settings)
            iban_to_use = iban_from_profile or (
                billing_settings.iban if billing_settings else None
            )

            if not iban_to_use:
                app_logger.warning(
                    "Pas d'IBAN configuré pour company_id=%s (ni profil ni settings)",
                    invoice.company_id,
                )
                return None

            # Créer le QR-Bill avec la vraie bibliothèque qrbill
            qr_bill = QRBill(
                account=iban_to_use,
                creditor=creditor_data,
                debtor={
                    "name": debtor_name,
                    "street": debtor_street,
                    "pcode": debtor_pcode,
                    "city": debtor_city,
                    "country": "CH",
                },
                amount=str(invoice.total_amount),
                currency="CHF",
                reference_number=self._get_payment_reference(invoice),
                additional_information=(
                    f"Facture {invoice.invoice_number} - "
                    f"Période: {invoice.period_month:02d}."
                    f"{invoice.period_year}"
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

                app_logger.info(
                    "QR-Bill SVG généré pour facture %s", invoice.invoice_number
                )
                return svg_content.encode("utf-8")

        except Exception as e:
            app_logger.error("Erreur lors de la génération du QR-Bill SVG: %s", str(e))
            return None

    def generate_qr_bill(self, invoice):
        """Génère un QR-Bill pour une facture."""
        try:
            # Récupérer les paramètres de facturation
            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=invoice.company_id
            ).first()

            # Récupérer les informations de la facture
            company = invoice.company
            client = invoice.client

            # Débiteur : Institution (si facturation tierce) ou Client
            # (si facturation directe)
            if (
                invoice.bill_to_client_id
                and invoice.bill_to_client_id != invoice.client_id
            ):
                # 🏥 Facturation tierce : débiteur = institution payeuse
                from models import Client as ClientModel

                institution = ClientModel.query.get(invoice.bill_to_client_id)

                if institution and institution.is_institution:
                    debtor_name = institution.institution_name or "Institution"
                    debtor_street = (
                        institution.billing_address
                        or institution.contact_address
                        or "Adresse non renseignée"
                    )
                    # Extraire code postal et ville de l'adresse si possible
                    debtor_pcode = "1200"
                    debtor_city = "Genève"
                else:
                    debtor_name = "Institution"
                    debtor_street = "Adresse non renseignée"
                    debtor_pcode = "1200"
                    debtor_city = "Genève"
            else:
                # 👤 Facturation directe : débiteur = client
                # (avec même logique que le PDF)
                debtor_name = (
                    (
                        f"{client.user.first_name or ''} {client.user.last_name or ''}"
                    ).strip()
                    or client.user.username
                    or "Client"
                )

                # Récupérer l'adresse avec priorités multiples
                debtor_street = "Adresse non renseignée"
                debtor_pcode = "1200"
                debtor_city = "Genève"

                # Priorité 1: Adresse du domicile
                if hasattr(client, "domicile_address") and client.domicile_address:
                    debtor_street = client.domicile_address
                    if hasattr(client, "domicile_zip") and client.domicile_zip:
                        debtor_pcode = client.domicile_zip
                    if hasattr(client, "domicile_city") and client.domicile_city:
                        debtor_city = client.domicile_city
                # Priorité 2: Adresse de l'utilisateur
                elif (
                    hasattr(client, "user")
                    and client.user
                    and hasattr(client.user, "address")
                    and client.user.address
                ):
                    full_address = client.user.address
                    # Format: "Allée de la Pépinière, 41, 74160, Archamps, France"
                    parts = [p.strip() for p in full_address.split(",")]
                    if len(parts) >= MIN_ADDRESS_PARTS:
                        # Rue + numéro
                        debtor_street = f"{parts[0]}, {parts[1]}"
                    if len(parts) >= MIN_ADDRESS_PARTS_POSTAL:
                        # Code postal
                        debtor_pcode = parts[2]
                    if len(parts) >= MIN_ADDRESS_PARTS_CITY:
                        # Ville
                        debtor_city = parts[3]

            # ✅ Utiliser l'adresse de domiciliation (cohérence avec PDF)
            creditor_info = self._get_creditor_info(company)
            creditor_data = creditor_info["address"]
            iban_from_profile = creditor_info["iban"]

            # Utiliser l'IBAN du profil en priorité (fallback sur billing_settings)
            iban_to_use = iban_from_profile or (
                billing_settings.iban if billing_settings else None
            )

            if not iban_to_use:
                app_logger.warning(
                    "Pas d'IBAN configuré pour company_id=%s (ni profil ni settings)",
                    invoice.company_id,
                )
                return None

            # Créer le QR-Bill avec la vraie bibliothèque qrbill
            qr_bill = QRBill(
                account=iban_to_use,
                creditor=creditor_data,
                debtor={
                    "name": debtor_name,
                    "street": debtor_street,
                    "pcode": debtor_pcode,
                    "city": debtor_city,
                    "country": "CH",
                },
                amount=str(invoice.total_amount),
                currency="CHF",
                reference_number=self._get_payment_reference(invoice),
                additional_information=(
                    f"Facture {invoice.invoice_number} - "
                    f"Période: {invoice.period_month:02d}."
                    f"{invoice.period_year}"
                ),
                language="fr",
            )

            # Générer le PDF du QR-Bill
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".svg", delete=False
            ) as temp_svg:
                qr_bill.as_svg(temp_svg.name)

                # Convertir SVG en PDF
                drawing = svg2rlg(temp_svg.name)

                # Créer le PDF en mémoire
                if drawing is None:
                    app_logger.error("Impossible de convertir le SVG en drawing")
                    return None

                pdf_buffer = BytesIO()
                renderPDF.drawToFile(drawing, pdf_buffer)
                pdf_buffer.seek(0)

                # Nettoyer le fichier temporaire
                Path(temp_svg.name).unlink()

                app_logger.info(
                    "QR-Bill généré pour facture %s", invoice.invoice_number
                )
                return pdf_buffer.getvalue()

        except Exception as e:
            app_logger.error("Erreur lors de la génération du QR-Bill: %s", str(e))
            return None

    def generate_qr_reference(self, invoice):
        """Génère une référence QR pour une facture."""
        try:
            # Générer une référence QR basée sur l'ID de la facture
            # Format: 27 caractères (modulo 10) - doit commencer par "RF"
            invoice_id_str = str(invoice.id).zfill(7)
            qr_reference = f"RF{invoice_id_str}"

            # Calculer le check digit (modulo 10)
            check_digit = self._calculate_check_digit(qr_reference)
            qr_reference += str(check_digit)

            # S'assurer que la référence fait exactement 27 caractères
            while len(qr_reference) < QR_REFERENCE_LENGTH:
                qr_reference += "0"

            return qr_reference[:QR_REFERENCE_LENGTH]  # Limiter à 27 caractères

        except Exception as e:
            app_logger.error(
                "Erreur lors de la génération de la référence QR: %s", str(e)
            )
            return None

    def _calculate_check_digit(self, reference):
        """Calcule le check digit pour une référence QR."""
        # Algorithme modulo 10 pour les références QR
        weights = [
            1,
            3,
            1,
            3,
            1,
            3,
            1,
            3,
            1,
            3,
            1,
            3,
            1,
            3,
            1,
            3,
            1,
            3,
            1,
            3,
            1,
            3,
            1,
            3,
            1,
            3,
        ]

        total = 0
        for i, char in enumerate(reference):
            if char.isdigit():
                total += int(char) * weights[i % len(weights)]
            else:
                # Pour les lettres, utiliser leur valeur ASCII
                total += (ord(char) - ord("A") + 10) * weights[i % len(weights)]

        remainder = total % 10
        return (10 - remainder) % 10
