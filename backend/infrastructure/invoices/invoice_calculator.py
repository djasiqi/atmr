"""Service d'infrastructure pour les calculs de facturation."""

from __future__ import annotations

import logging
from decimal import ROUND_HALF_UP, Decimal
from typing import Protocol

from domain.invoice_dto import InvoiceLineDTO

logger = logging.getLogger(__name__)


def round_to_5_cents(amount: Decimal) -> Decimal:
    """Arrondit un montant à 5 centimes (0.05 CHF).

    Exemples:
        - 10.12 → 10.10
        - 10.13 → 10.15
        - 10.17 → 10.15
        - 10.18 → 10.20
        - 11.13 → 11.15

    Args:
        amount: Montant à arrondir

    Returns:
        Montant arrondi à 5 centimes
    """
    # Multiplier par 20 pour convertir en multiples de 5 centimes
    # Arrondir à l'entier le plus proche
    # Diviser par 20 pour revenir aux francs
    rounded = (amount * Decimal("20")).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    ) / Decimal("20")
    # S'assurer que le résultat a 2 décimales
    return rounded.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class InvoiceCalculatorPort(Protocol):
    """Port (interface) pour le calculateur de facturation."""

    def calculate_vat(
        self, base_amount: Decimal, vat_rate: Decimal
    ) -> tuple[Decimal, Decimal]:
        """Calcule la TVA et le total avec TVA.

        Args:
            base_amount: Montant de base (HT)
            vat_rate: Taux de TVA (ex: 7.7 pour 7.7%)

        Returns:
            Tuple (vat_amount, total_with_vat)
        """
        ...

    def calculate_totals(
        self, lines: list[InvoiceLineDTO]
    ) -> tuple[Decimal, Decimal, Decimal, dict[str, dict[str, Decimal]]]:
        """Calcule les totaux d'une facture.

        Args:
            lines: Liste des lignes de facture

        Returns:
            Tuple (subtotal, vat_total, total, vat_breakdown)
            où vat_breakdown est un dict {rate: {base, vat}}
        """
        ...


class InvoiceCalculator:
    """Calculateur de facturation."""

    def calculate_vat(
        self, base_amount: Decimal, vat_rate: Decimal
    ) -> tuple[Decimal, Decimal]:
        """Calcule la TVA et le total avec TVA.

        Args:
            base_amount: Montant de base (HT)
            vat_rate: Taux de TVA (ex: 7.7 pour 7.7%)

        Returns:
            Tuple (vat_amount, total_with_vat)
        """
        two_places = Decimal("0.01")
        vat_amount = (base_amount * vat_rate / Decimal("100")).quantize(
            two_places, rounding=ROUND_HALF_UP
        )
        total_with_vat = (base_amount + vat_amount).quantize(
            two_places, rounding=ROUND_HALF_UP
        )
        # Arrondir le total à 5 centimes pour éviter les montants comme 10.12 ou 11.13
        total_with_vat = round_to_5_cents(total_with_vat)
        # Recalculer la TVA pour qu'elle corresponde au total arrondi
        vat_amount = total_with_vat - base_amount
        # S'assurer que la TVA est positive et arrondie correctement
        if vat_amount < 0:
            vat_amount = Decimal("0.00")
        else:
            vat_amount = vat_amount.quantize(two_places, rounding=ROUND_HALF_UP)
        return vat_amount, total_with_vat

    def calculate_totals(
        self, lines: list[InvoiceLineDTO]
    ) -> tuple[Decimal, Decimal, Decimal, dict[str, dict[str, Decimal]]]:
        """Calcule les totaux d'une facture.

        Args:
            lines: Liste des lignes de facture

        Returns:
            Tuple (subtotal, vat_total, total, vat_breakdown)
            où vat_breakdown est un dict {rate: {base, vat}}
        """
        two_places = Decimal("0.01")
        subtotal = Decimal("0.00")
        vat_total = Decimal("0.00")
        vat_breakdown: dict[str, dict[str, Decimal]] = {}

        for line in lines:
            base_amount = line.line_total
            subtotal += base_amount

            if line.vat_rate and line.vat_rate > Decimal("0"):
                vat_amount = line.vat_amount
                vat_total += vat_amount
                rate_key = f"{line.vat_rate.normalize()}"
                if rate_key not in vat_breakdown:
                    vat_breakdown[rate_key] = {
                        "base": Decimal("0.00"),
                        "vat": Decimal("0.00"),
                    }
                vat_breakdown[rate_key]["base"] += base_amount
                vat_breakdown[rate_key]["vat"] += vat_amount

        total = (subtotal + vat_total).quantize(two_places, rounding=ROUND_HALF_UP)
        # Arrondir le total à 5 centimes pour éviter les montants comme 10.12 ou 11.13
        total = round_to_5_cents(total)
        # Ajuster la TVA totale pour qu'elle corresponde au total arrondi
        vat_total = total - subtotal
        # S'assurer que la TVA totale est positive et arrondie correctement
        if vat_total < 0:
            vat_total = Decimal("0.00")
        else:
            vat_total = vat_total.quantize(two_places, rounding=ROUND_HALF_UP)

        return subtotal, vat_total, total, vat_breakdown
