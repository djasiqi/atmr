import logging
import re
import tempfile
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
from typing import Any

from qrbill import QRBill
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

from infrastructure.invoices.invoice_calculator import round_to_5_cents
from services.billing import BillingProfileService, generate_scor_reference

QR_REFERENCE_LENGTH = 27
QRR_BASE_LENGTH = 2  # Longueur de creditor_reference_base (ex: "21")
QRR_INVOICE_NUM_LENGTH = 20  # Longueur max pour partie invoice_number
QRR_INVOICE_ID_LENGTH = 4  # Longueur max pour invoice.id
QRR_REF_BASE_LENGTH = 26  # Longueur base avant check digit
QRR_MIN_IBAN_LENGTH = 5  # Longueur minimale IBAN pour validation

app_logger = logging.getLogger("qrbill_service")

# Tolérance (arrondis 5 c.) entre Σ lignes TTC et total_amount facture
_QR_LINE_TOTAL_TOLERANCE = Decimal("0.05")


def _format_amount_for_qrbill(amount: Decimal) -> str:
    """Format fixe 2 décimales pour la lib qrbill / norme paiement CH."""
    quantized = amount.quantize(Decimal("0.01"))
    return f"{quantized:.2f}"


def resolve_qr_bill_amount_decimal(
    invoice: Any, override_amount: Any | None = None
) -> Decimal:
    """Montant à encoder sur le QR-facture (CHF), aligné sur ce qui est dû.

    - Rappels / cas spéciaux : ``override_amount`` (ex. total dû rappel + frais).
    - Sinon : ``balance_due`` (solde après acomptes), sinon ``total_amount``.

    Arrondi **5 centimes** (0,05 CHF) :
    - toujours pour ``override_amount`` (rappels) ;
    - pour le montant facture / solde **tant qu'aucun acompte** n'a été enregistré
      (sinon le solde partiel peut légitimement finir en centimes « non multiples de 5 »).

    À la génération, ``total_amount`` est déjà arrondi 5 c. ; cet arrondi renforce
    la cohérence QR / total affiché si des données legacy sont au centime « cassé ».
    """
    two = Decimal("0.01")
    if override_amount is not None:
        try:
            d = Decimal(str(override_amount))
        except (InvalidOperation, TypeError, ValueError):
            app_logger.warning(
                "[QR-Bill] override_amount invalide %r, repli sur facture",
                override_amount,
            )
            d = Decimal("0.00")
        out = max(d.quantize(two), Decimal("0.00"))
        return max(round_to_5_cents(out), Decimal("0.00"))

    out: Decimal | None = None
    for attr in ("balance_due", "total_amount"):
        raw = getattr(invoice, attr, None)
        if raw is None:
            continue
        try:
            out = max(Decimal(str(raw)).quantize(two), Decimal("0.00"))
            break
        except (InvalidOperation, TypeError, ValueError):
            continue
    if out is None:
        return Decimal("0.00")

    paid_raw = getattr(invoice, "amount_paid", None)
    try:
        paid_d = Decimal(str(paid_raw if paid_raw is not None else 0)).quantize(two)
    except (InvalidOperation, TypeError, ValueError):
        paid_d = Decimal("0.00")

    if paid_d <= Decimal("0.00"):
        out = max(round_to_5_cents(out), Decimal("0.00"))
    return out


def warn_if_invoice_line_totals_mismatch_invoice_total(invoice: Any) -> None:
    """Journalise un avertissement si Σ total_with_vat des lignes ≠ total_amount."""
    if getattr(invoice, "id", None) is None:
        return
    lines = getattr(invoice, "lines", None) or []
    if not lines:
        return
    total_ref = getattr(invoice, "total_amount", None)
    if total_ref is None:
        return
    try:
        sum_ttc = sum(
            (Decimal(str(ln.total_with_vat or 0)) for ln in lines),
            Decimal("0.00"),
        ).quantize(Decimal("0.01"))
        ref_d = Decimal(str(total_ref)).quantize(Decimal("0.01"))
    except (InvalidOperation, TypeError, ValueError):
        return
    if abs(sum_ttc - ref_d) > _QR_LINE_TOTAL_TOLERANCE:
        app_logger.warning(
            "[QR-Bill] Cohérence facture: somme des lignes TTC=%s vs "
            "invoice.total_amount=%s (invoice_id=%s). Le montant QR est basé sur "
            "balance_due/total_amount — vérifier les lignes et remises.",
            sum_ttc,
            ref_d,
            getattr(invoice, "id", "?"),
        )


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
                    if (
                        not qr_iban.startswith("CH")
                        or len(qr_iban) < QRR_MIN_IBAN_LENGTH
                    ):
                        error_msg = (
                            f"QR-IBAN invalide pour mode QRR "
                            f"(len={len(qr_iban)}, prefix_ok={qr_iban.startswith('CH')}). "
                            f"Un QR-IBAN doit commencer par 'CH' et avoir au moins 5 caractères. "
                            f"Veuillez configurer un QR-IBAN valide (format CH..3…)."
                        )
                        app_logger.error(
                            "[QR-Bill] QR-IBAN invalide company_id=%s invoice=%s len=%s prefix_ok=%s",
                            invoice.company_id,
                            invoice.invoice_number,
                            len(qr_iban),
                            qr_iban.startswith("CH"),
                        )
                        raise ValueError(error_msg)

                    if qr_iban[4:5] != "3":
                        error_msg = (
                            "QR-IBAN invalide pour mode QRR: "
                            "Le 5ème caractère doit être '3' (QR-IBAN requis). "
                            "Veuillez configurer un QR-IBAN valide (format CH..3…)."
                        )
                        app_logger.error(
                            "[QR-Bill] QR-IBAN non-QRR company_id=%s invoice=%s",
                            invoice.company_id,
                            invoice.invoice_number,
                        )
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

        except ValueError:
            raise
        except Exception as e:
            app_logger.error("[QR-Bill] Erreur génération référence : %s", e)
            result = None

        return result

    def _parse_address_for_qrbill(self, address: str) -> tuple[str, str, str]:
        """Parse une adresse pour QR-bill en séparant rue, code postal et ville."""
        from services.documents.qr_debtor import parse_address_for_qrbill

        return parse_address_for_qrbill(address)

    def _get_debtor_info(self, invoice) -> dict[str, Any]:
        """Débiteur « Payable par » : snapshot si facture figée, sinon règles live."""
        from services.documents.qr_debtor import resolve_invoice_qr_debtor

        return resolve_invoice_qr_debtor(invoice).to_qrbill_dict()

    def _get_debtor_for_institution_patient(self, invoice, client) -> dict[str, Any]:
        """Compatibilité tests / appels internes : débiteur S1 institution live."""
        from services.documents.qr_debtor import _live_institution_patient

        return _live_institution_patient(invoice, client).to_qrbill_dict()

    def _get_creditor_info(self, company, invoice=None):
        """Créancier QR : snapshot si facture figée, sinon master data live."""
        from services.documents.qr_creditor import (
            resolve_invoice_qr_creditor,
            resolve_qr_creditor_live,
        )

        if invoice is not None:
            creditor = resolve_invoice_qr_creditor(invoice)
        else:
            proxy = type(
                "InvoiceProxy", (), {"company": company, "company_id": company.id}
            )()
            creditor = resolve_qr_creditor_live(proxy)
        return {
            "address": creditor.to_qrbill_address(),
            "iban": creditor.account or None,
            "address_type": creditor.address_type,
        }

    def _resolve_qr_account_and_creditor(self, invoice):
        from services.documents.qr_creditor import resolve_invoice_qr_creditor

        creditor = resolve_invoice_qr_creditor(invoice)
        return creditor.account or None, creditor.to_qrbill_address()

    def generate_qr_bill_svg(self, invoice, override_amount=None):
        """Génère un QR-Bill SVG pour une facture.

        Args:
            invoice: Facture pour laquelle générer le QR-Bill.
            override_amount: Montant à utiliser à la place de invoice.total_amount
                (ex. total rappel incluant les frais).
        """
        try:
            debtor_data = self._get_debtor_info(invoice)
            iban_to_use, creditor_data = self._resolve_qr_account_and_creditor(invoice)

            if not iban_to_use:
                app_logger.warning(
                    "Pas d'IBAN configuré pour company_id=%s (ni profil ni settings)",
                    invoice.company_id,
                )
                return None

            # Créer le QR-Bill avec la vraie bibliothèque qrbill
            qr_amount = (
                str(override_amount)
                if override_amount is not None
                else str(invoice.total_amount)
            )
            qr_bill = QRBill(
                account=iban_to_use,
                creditor=creditor_data,
                debtor=debtor_data,
                amount=qr_amount,
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
            debtor_data = self._get_debtor_info(invoice)
            iban_to_use, creditor_data = self._resolve_qr_account_and_creditor(invoice)

            if not iban_to_use:
                app_logger.warning(
                    "Pas d'IBAN configuré pour company_id=%s (ni profil ni settings)",
                    invoice.company_id,
                )
                return None

            warn_if_invoice_line_totals_mismatch_invoice_total(invoice)
            qr_dec = resolve_qr_bill_amount_decimal(invoice, None)
            qr_amount = _format_amount_for_qrbill(qr_dec)
            app_logger.debug(
                "[QR-Bill] Montant encodé (CHF)=%s pour facture %s",
                qr_amount,
                getattr(invoice, "invoice_number", "?"),
            )

            # Créer le QR-Bill avec la vraie bibliothèque qrbill
            qr_bill = QRBill(
                account=iban_to_use,
                creditor=creditor_data,
                debtor=debtor_data,
                amount=qr_amount,
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
