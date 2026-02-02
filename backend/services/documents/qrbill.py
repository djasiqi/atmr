import logging
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

from qrbill import QRBill
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

from models import CompanyBillingSettings
from services.billing import BillingProfileService, generate_scor_reference

# Constantes pour éviter les valeurs magiques
MIN_ADDRESS_PARTS = 2
MIN_ADDRESS_PARTS_POSTAL = 3
MIN_ADDRESS_PARTS_CITY = 4
QR_REFERENCE_LENGTH = 27
QRR_BASE_LENGTH = 2  # Longueur de creditor_reference_base (ex: "21")
QRR_INVOICE_NUM_LENGTH = 20  # Longueur max pour partie invoice_number
QRR_INVOICE_ID_LENGTH = 4  # Longueur max pour invoice.id
QRR_REF_BASE_LENGTH = 26  # Longueur base avant check digit
QRR_MIN_IBAN_LENGTH = 5  # Longueur minimale IBAN pour validation

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
        result = None
        try:
            # ✅ Réutiliser si déjà généré (stabilité)
            if invoice.qr_reference:
                app_logger.debug(
                    "[QR-Bill] Réutilisation qr_reference existante: %s",
                    invoice.qr_reference,
                )
                result = invoice.qr_reference
            else:
                # Récupérer le profil de facturation
                profile = BillingProfileService.get_by_company_id(invoice.company_id)

                if not profile:
                    app_logger.warning(
                        "[QR-Bill] Pas de profil pour company_id=%s, pas de référence générée",
                        invoice.company_id,
                    )
                    result = None
                elif profile.payment_reference_mode == "NONE":
                    app_logger.debug("[QR-Bill] Mode NONE : pas de référence")
                    result = None
                elif profile.payment_reference_mode == "SCOR":
                    # Générer une référence SCOR (ISO 11649)
                    app_logger.debug(
                        "[QR-Bill] Génération SCOR pour %s", invoice.invoice_number
                    )
                    result = generate_scor_reference(
                        invoice.invoice_number, company_id=invoice.company_id
                    )
                elif profile.payment_reference_mode == "QRR":
                    # ✅ Valider QR-IBAN (CH..3…) - lever exception si invalide
                    qr_iban = profile.qr_iban or profile.iban
                    if not qr_iban:
                        error_msg = (
                            f"Mode QRR nécessite un QR-IBAN. "
                            f"Company {invoice.company_id} n'a pas de qr_iban configuré. "
                            f"Veuillez configurer un QR-IBAN valide (format CH..3…) dans les paramètres de facturation."
                        )
                        app_logger.error("[QR-Bill] %s", error_msg)
                        raise ValueError(error_msg)

                    # Vérifier format QR-IBAN (CH..3…)
                    if not qr_iban.startswith("CH") or len(qr_iban) < QRR_MIN_IBAN_LENGTH:
                        error_msg = (
                            f"QR-IBAN invalide pour mode QRR: {qr_iban}. "
                            f"Un QR-IBAN doit commencer par 'CH' et avoir au moins 5 caractères. "
                            f"Veuillez configurer un QR-IBAN valide (format CH..3…)."
                        )
                        app_logger.error("[QR-Bill] %s", error_msg)
                        raise ValueError(error_msg)

                    if qr_iban[4:5] != "3":
                        error_msg = (
                            f"QR-IBAN invalide pour mode QRR: {qr_iban}. "
                            f"Le 5ème caractère doit être '3' (QR-IBAN requis). "
                            f"Veuillez configurer un QR-IBAN valide (format CH..3…)."
                        )
                        app_logger.error("[QR-Bill] %s", error_msg)
                        raise ValueError(error_msg)

                    # ✅ Générer référence QRR (27 chiffres numériques)
                    app_logger.debug(
                        "[QR-Bill] Génération QRR pour %s", invoice.invoice_number
                    )
                    result = self._generate_qrr_reference(invoice, profile)
                else:
                    app_logger.error(
                        "[QR-Bill] Mode de référence inconnu : %s",
                        profile.payment_reference_mode,
                    )
                    result = None

        except Exception as e:
            app_logger.error("[QR-Bill] Erreur génération référence : %s", e)
            result = None

        return result

    def _parse_address_for_qrbill(self, address: str) -> tuple[str, str, str]:
        """Parse une adresse pour QR-bill en séparant rue, code postal et ville."""
        if not address:
            return ("", "1200", "Genève")
        parts = [p.strip() for p in address.replace("\n", ",").split(",")]
        # Format "Rue, Numéro, CP, Ville" ou "Rue Numéro, CP Ville"
        if len(parts) >= MIN_ADDRESS_PARTS_CITY:
            street = f"{parts[0]}, {parts[1]}" if len(parts) > 1 else parts[0]
            return (street, parts[2], parts[3])
        if len(parts) >= MIN_ADDRESS_PARTS_POSTAL:
            # "Rue, CP Ville" ou "Rue Numéro, CP, Ville"
            street = parts[0]
            pcode_city = parts[1].strip().split()
            if len(pcode_city) >= MIN_ADDRESS_PARTS:
                return (street, pcode_city[0], " ".join(pcode_city[1:]))
            return (
                street,
                parts[1],
                parts[2] if len(parts) >= MIN_ADDRESS_PARTS_POSTAL else "Genève",
            )
        if len(parts) >= MIN_ADDRESS_PARTS:
            last_part = parts[-1].strip().split()
            if len(last_part) >= MIN_ADDRESS_PARTS:
                return (parts[0], last_part[0], " ".join(last_part[1:]))
        return (address, "1200", "Genève")

    def _get_debtor_info(self, invoice) -> dict[str, Any]:
        """Résout le débiteur (Payable par) pour le QR-bill."""
        client = invoice.client

        # S2 facture clinique mensuelle : débiteur = clinique
        strategy_val = (
            getattr(invoice.billing_strategy, "value", None)
            if invoice.billing_strategy
            else None
        ) or str(getattr(invoice, "billing_strategy", "") or "")
        if strategy_val == "s2_clinic_monthly" and getattr(
            invoice, "billed_to_company_id", None
        ):
            debtor_name = "Clinique"
            debtor_street = "Adresse non renseignée"
            debtor_pcode = "1200"
            debtor_city = "Genève"

            bp = getattr(invoice, "billing_party", None)
            if bp is not None:
                debtor_name = (getattr(bp, "display_name", None) or "Clinique").strip()
                addr = (getattr(bp, "billing_address", None) or "").strip()
                if addr:
                    debtor_street, debtor_pcode, debtor_city = (
                        self._parse_address_for_qrbill(addr)
                    )
            else:
                clinic = getattr(invoice, "billed_to_company", None)
                if clinic is not None:
                    debtor_name = (getattr(clinic, "name", None) or "Clinique").strip()
                    line1 = (getattr(clinic, "domicile_address_line1", None) or "").strip()
                    line2 = (getattr(clinic, "domicile_address_line2", None) or "").strip()
                    debtor_street = f"{line1} {line2}".strip() or "Adresse non renseignée"
                    debtor_pcode = getattr(clinic, "domicile_zip", None) or "1200"
                    debtor_city = getattr(clinic, "domicile_city", None) or "Genève"

            return {
                "name": debtor_name,
                "street": debtor_street,
                "pcode": debtor_pcode,
                "city": debtor_city,
                "country": "CH",
            }

        # Facturation tierce : institution (bill_to_client_id)
        if (
            invoice.bill_to_client_id
            and invoice.bill_to_client_id != invoice.client_id
        ):
            from models import Client as ClientModel

            institution = ClientModel.query.get(invoice.bill_to_client_id)
            if institution and institution.is_institution:
                debtor_name = institution.institution_name or "Institution"
                debtor_street = (
                    institution.billing_address
                    or institution.contact_address
                    or "Adresse non renseignée"
                )
            else:
                debtor_name = "Institution"
                debtor_street = "Adresse non renseignée"
            return {
                "name": debtor_name,
                "street": debtor_street,
                "pcode": "1200",
                "city": "Genève",
                "country": "CH",
            }

        # Facturation directe : client
        debtor_name = (
            (
                f"{client.user.first_name or ''} {client.user.last_name or ''}"
            ).strip()
            or client.user.username
            or "Client"
        )
        debtor_street = "Adresse non renseignée"
        debtor_pcode = "1200"
        debtor_city = "Genève"
        if hasattr(client, "domicile_address") and client.domicile_address:
            debtor_street = client.domicile_address
            if hasattr(client, "domicile_zip") and client.domicile_zip:
                debtor_pcode = client.domicile_zip
            if hasattr(client, "domicile_city") and client.domicile_city:
                debtor_city = client.domicile_city
        elif (
            hasattr(client, "user")
            and client.user
            and hasattr(client.user, "address")
            and client.user.address
        ):
            parts = [p.strip() for p in client.user.address.split(",")]
            if len(parts) >= MIN_ADDRESS_PARTS:
                debtor_street = f"{parts[0]}, {parts[1]}"
            if len(parts) >= MIN_ADDRESS_PARTS_POSTAL:
                debtor_pcode = parts[2]
            if len(parts) >= MIN_ADDRESS_PARTS_CITY:
                debtor_city = parts[3]

        return {
            "name": debtor_name,
            "street": debtor_street,
            "pcode": debtor_pcode,
            "city": debtor_city,
            "country": "CH",
        }

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

            # Débiteur : S2 clinique, institution tierce ou client direct
            debtor_data = self._get_debtor_info(invoice)

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
                debtor=debtor_data,
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

            # Débiteur : S2 clinique, institution tierce ou client direct
            debtor_data = self._get_debtor_info(invoice)

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
                debtor=debtor_data,
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

    def _calculate_qrr_check_digit(self, reference_base: str) -> int:
        """Calcule le check digit QRR avec l'algorithme modulo 10 récursif (ISO 7064).

        Args:
            reference_base: 26 chiffres numériques (sans check digit)

        Returns:
            int: Check digit (0-9)
        """
        # Algorithme modulo 10 récursif (ISO 7064 MOD 10, RECURSIVE)
        # Accumulateur initial = 10
        accumulator = 10

        for digit_char in reference_base:
            if not digit_char.isdigit():
                raise ValueError(f"QRR reference doit être numérique: {reference_base}")
            digit = int(digit_char)
            accumulator = (accumulator + digit) % 10
            if accumulator == 0:
                accumulator = 10

        # Check digit = (10 - accumulator) % 10
        return (10 - accumulator) % 10

    def _generate_qrr_reference(self, invoice, profile) -> str:
        """Génère une référence QRR (ESR) de 27 chiffres numériques.

        Format: Base (creditor_reference_base) + invoice_number + invoice.id + check digit
        Exemple: 210000000000000000000123456

        Args:
            invoice: Facture pour laquelle générer la référence
            profile: Profil de facturation (CompanyBillingProfile)

        Returns:
            str: Référence QRR de 27 chiffres (numérique uniquement)

        Raises:
            ValueError: Si la référence ne peut pas être générée correctement
        """
        # Utiliser creditor_reference_base si disponible (ex: "21")
        # Sinon, utiliser "21" par défaut (code standard suisse)
        base = profile.creditor_reference_base or "21"

        # ✅ Normaliser invoice_number : extraire uniquement les chiffres
        invoice_num_digits = re.sub(r"\D", "", invoice.invoice_number)

        if not invoice_num_digits:
            raise ValueError(
                f"Impossible de générer QRR : invoice_number '{invoice.invoice_number}' "
                + "ne contient aucun chiffre"
            )

        # ✅ Garantir unicité : ajouter invoice.id pour éviter collisions
        # Format: base (2) + invoice_num (max 20) + invoice.id (max 4) = 26 chiffres
        # On prend les 20 derniers chiffres de invoice_number pour laisser place à invoice.id
        invoice_num_part = (
            invoice_num_digits[-QRR_INVOICE_NUM_LENGTH:]
            if len(invoice_num_digits) > QRR_INVOICE_NUM_LENGTH
            else invoice_num_digits
        )
        invoice_id_str = str(invoice.id)

        # Construire la base : base (2) + invoice_num (20) + invoice.id (4) = 26 chiffres
        # Si invoice.id est trop long, on tronque
        if len(invoice_id_str) > QRR_INVOICE_ID_LENGTH:
            app_logger.warning(
                "[QR-Bill] invoice.id trop long (%s > %s), troncature pour QRR",
                len(invoice_id_str),
                QRR_INVOICE_ID_LENGTH,
            )
            invoice_id_str = invoice_id_str[-QRR_INVOICE_ID_LENGTH:]

        # Construire la base de référence (26 chiffres pour le check digit)
        ref_base = (
            base
            + invoice_num_part.rjust(QRR_INVOICE_NUM_LENGTH, "0")
            + invoice_id_str.zfill(QRR_INVOICE_ID_LENGTH)
        )

        # Vérifier la longueur (doit être exactement 26)
        if len(ref_base) != QRR_REF_BASE_LENGTH:
            # Ajuster si nécessaire
            ref_base = (
                ref_base[:QRR_REF_BASE_LENGTH]
                if len(ref_base) > QRR_REF_BASE_LENGTH
                else ref_base.ljust(QRR_REF_BASE_LENGTH, "0")
            )

        # Calculer le check digit (modulo 10 récursif) et construire la référence complète
        qrr_reference = ref_base + str(self._calculate_qrr_check_digit(ref_base))

        # Validation finale
        if len(qrr_reference) != QR_REFERENCE_LENGTH:
            raise ValueError(
                "Erreur génération QRR : longueur incorrecte "
                + f"({len(qrr_reference)} != {QR_REFERENCE_LENGTH})"
            )

        if not qrr_reference.isdigit():
            raise ValueError(
                f"Erreur génération QRR : référence contient des caractères non numériques: {qrr_reference}"
            )

        app_logger.debug(
            "[QR-Bill] QRR générée: %s (base: %s, invoice: %s, id: %s)",
            qrr_reference,
            base,
            invoice.invoice_number,
            invoice.id,
        )

        return qrr_reference
