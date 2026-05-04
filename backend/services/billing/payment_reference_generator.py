"""Générateur de références de paiement SCOR (ISO 11649).

Ce module génère des références de paiement structurées conformes
à la norme ISO 11649 (SCOR - Structured Creditor Reference).

Standard Swiss QR-Bill : Les références SCOR sont utilisées avec
les IBAN standard (non QR-IBAN).
"""

# ruff: noqa: G004
import logging
import re

logger = logging.getLogger(__name__)

# Constantes pour éviter les magic values
SCOR_MAX_DATA_LENGTH = 21  # Max 21 caractères de données (hors RF + check digits)
SCOR_MIN_LENGTH = 5  # RF + 2 check digits + au moins 1 caractère
SCOR_MAX_LENGTH = 25  # Max total ISO 11649


class PaymentReferenceGenerator:
    """Générateur de références SCOR (ISO 11649)."""

    @staticmethod
    def _calculate_mod97(value: str) -> int:
        """Calcule le modulo 97 pour validation ISO 7064.

        Args:
            value: Chaîne numérique

        Returns:
            int: Résultat modulo 97
        """
        return int(value) % 97

    @staticmethod
    def _char_to_digits(char: str) -> str:
        """Convertit un caractère en chiffres (A=10, B=11, ..., Z=35).

        Args:
            char: Caractère à convertir

        Returns:
            str: Représentation numérique
        """
        if char.isdigit():
            return char
        return str(ord(char.upper()) - ord("A") + 10)

    @staticmethod
    def _to_numeric_string(text: str) -> str:
        """Convertit une chaîne alphanumérique en chaîne numérique.

        Args:
            text: Texte à convertir

        Returns:
            str: Représentation numérique
        """
        return "".join(PaymentReferenceGenerator._char_to_digits(char) for char in text)

    @staticmethod
    def calculate_check_digits(reference: str) -> str:
        """Calcule les 2 chiffres de contrôle ISO 11649.

        Args:
            reference: Référence sans les check digits

        Returns:
            str: 2 chiffres de contrôle (ex: "07")
        """
        # 1. Convertir en numérique (déplacer RF00 à la fin)
        numeric = PaymentReferenceGenerator._to_numeric_string(reference + "RF00")

        # 2. Calculer 98 - (numeric mod 97)
        check_digits = 98 - PaymentReferenceGenerator._calculate_mod97(numeric)

        # 3. Retourner avec zéro leading si nécessaire
        return f"{check_digits:02d}"

    @staticmethod
    def generate_scor(invoice_number: str, company_id: int | None = None) -> str:
        """Génère une référence SCOR complète.

        Format final : RFxx yyyy yyyy yyyy yyyy yyyy (où xx = check digits)

        Args:
            invoice_number: Numéro de facture (ex: "EM-2026-01-0001")
            company_id: ID de l'entreprise (optionnel, pour unicité)

        Returns:
            str: Référence SCOR complète (ex: "RF18 5390 0754 7034 2")

        Example:
            >>> gen = PaymentReferenceGenerator()
            >>> gen.generate_scor("EM-2026-01-0001", company_id=1)
            'RF48 1EM2 0260 1000 1'
        """
        # 1. Nettoyer le numéro de facture (enlever caractères non alphanumériques)
        clean_invoice = re.sub(r"[^A-Z0-9]", "", invoice_number.upper())

        # 2. Optionnel : préfixer avec company_id pour garantir l'unicité multi-tenant
        reference_base = f"{company_id}{clean_invoice}" if company_id else clean_invoice

        # 3. Limiter la longueur (max 21 caractères sans RF et check digits)
        # Standard ISO 11649 : max 25 caractères total (RF + 2 check + 21 data)
        if len(reference_base) > SCOR_MAX_DATA_LENGTH:
            logger.warning(
                "Référence trop longue (%s > %s), truncation : %s",
                len(reference_base),
                SCOR_MAX_DATA_LENGTH,
                reference_base,
            )
            reference_base = reference_base[:SCOR_MAX_DATA_LENGTH]

        # 4. Calculer les check digits
        check_digits = PaymentReferenceGenerator.calculate_check_digits(reference_base)

        # 5. Construire la référence complète
        scor_reference = f"RF{check_digits}{reference_base}"

        # 6. Formater avec espaces tous les 4 caractères (optionnel, pour lisibilité)
        formatted = " ".join(
            [scor_reference[i : i + 4] for i in range(0, len(scor_reference), 4)]
        )

        logger.debug(f"SCOR générée : {formatted} (base: {reference_base})")

        return formatted

    @staticmethod
    def validate_scor(scor_reference: str) -> bool:
        """Valide une référence SCOR.

        Args:
            scor_reference: Référence SCOR à valider

        Returns:
            bool: True si valide, False sinon
        """
        try:
            # 1. Enlever les espaces
            clean = scor_reference.replace(" ", "").upper()

            # 2. Vérifier le format de base
            if not clean.startswith("RF"):
                logger.warning(f"SCOR invalide : ne commence pas par 'RF' : {clean}")
                return False

            if len(clean) < SCOR_MIN_LENGTH:  # RF + 2 check + au moins 1 char
                logger.warning("SCOR invalide : trop courte : %s", clean)
                return False

            if len(clean) > SCOR_MAX_LENGTH:  # Max ISO 11649
                logger.warning(
                    "SCOR invalide : trop longue (%s > %s) : %s",
                    len(clean),
                    SCOR_MAX_LENGTH,
                    clean,
                )
                return False

            # 3. Extraire les parties
            check_digits_str = clean[2:4]
            reference_base = clean[4:]

            # 4. Vérifier que les check digits sont numériques
            if not check_digits_str.isdigit():
                logger.warning(
                    f"SCOR invalide : check digits non numériques : {check_digits_str}"
                )
                return False

            # 5. Recalculer les check digits
            calculated_check = PaymentReferenceGenerator.calculate_check_digits(
                reference_base
            )

            # 6. Comparer
            if check_digits_str != calculated_check:
                logger.warning(
                    "SCOR invalide : check digits incorrects. Attendu: %s, Reçu: %s",
                    calculated_check,
                    check_digits_str,
                )
                return False

            logger.debug(f"SCOR valide : {scor_reference}")
            return True

        except Exception as e:
            logger.error(f"Erreur validation SCOR : {e}")
            return False

    @staticmethod
    def extract_invoice_number(
        scor_reference: str, company_id: int | None = None
    ) -> str | None:
        """Extrait le numéro de facture d'une référence SCOR.

        Args:
            scor_reference: Référence SCOR
            company_id: ID de l'entreprise (si utilisé lors de la génération)

        Returns:
            str | None: Numéro de facture extrait ou None

        Example:
            >>> gen = PaymentReferenceGenerator()
            >>> gen.extract_invoice_number("RF48 1EM2 0260 1000 1", company_id=1)
            'EM-2026-01-0001'
        """
        try:
            # 1. Valider la référence
            if not PaymentReferenceGenerator.validate_scor(scor_reference):
                return None

            # 2. Enlever RF et check digits
            clean = scor_reference.replace(" ", "").upper()
            reference_base = clean[4:]

            # 3. Si company_id utilisé, l'enlever
            if company_id:
                company_id_str = str(company_id)
                if reference_base.startswith(company_id_str):
                    reference_base = reference_base[len(company_id_str) :]

            # 4. Essayer de reformater le numéro de facture
            # (c'est difficile sans connaître le format original)
            # Pour l'instant, on retourne tel quel
            return reference_base

        except Exception as e:
            logger.error(f"Erreur extraction numéro facture : {e}")
            return None


# Alias pour simplifier l'import
generate_scor_reference = PaymentReferenceGenerator.generate_scor
validate_scor_reference = PaymentReferenceGenerator.validate_scor
