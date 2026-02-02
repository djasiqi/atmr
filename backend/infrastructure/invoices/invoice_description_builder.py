"""Helper pour construire les descriptions de lignes de facture."""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class InvoiceDescriptionBuilderPort(Protocol):
    """Port (interface) pour le constructeur de descriptions."""

    def build_description(
        self,
        pickup_location: str,
        dropoff_location: str,
        patient_name: str | None = None,
        bill_to_client_id: int | None = None,
        *,
        is_cancelled: bool = False,
    ) -> str:
        """Construit la description d'une ligne de facture.

        Args:
            pickup_location: Lieu de prise en charge
            dropoff_location: Lieu de dépose
            patient_name: Nom du patient (optionnel, pour facturation tierce)
            bill_to_client_id: ID du client payeur (optionnel, pour facturation tierce)

        Returns:
            Description formatée
        """
        ...


class InvoiceDescriptionBuilder:
    """Constructeur de descriptions de lignes de facture."""

    def build_description(
        self,
        pickup_location: str,
        dropoff_location: str,
        patient_name: str | None = None,
        bill_to_client_id: int | None = None,
        *,
        is_material_delivery: bool = False,
        delivery_description: str | None = None,
        is_cancelled: bool = False,
    ) -> str:
        """Construit la description d'une ligne de facture.

        Args:
            pickup_location: Lieu de prise en charge
            dropoff_location: Lieu de dépose
            patient_name: Nom du patient (optionnel, pour facturation tierce)
            bill_to_client_id: ID du client payeur (optionnel, pour facturation tierce)
            is_material_delivery: True si livraison matériel
            delivery_description: Description de la livraison (ex: Livraison médicament)
            is_cancelled: True si transport annulé (mention "Annulation" préfixée)

        Returns:
            Description formatée
        """
        prefix = "Annulation – " if is_cancelled else ""
        if is_material_delivery and delivery_description:
            # Livraison : "Livraison medicament - Clinique -> Domicile"
            return f"{prefix}Livraison – {delivery_description} – {pickup_location} → {dropoff_location}"
        if bill_to_client_id and patient_name:
            # Facturation tierce : inclure le nom du patient
            return f"{prefix}Trajet pour {patient_name}: {pickup_location} → {dropoff_location}"
        # Facturation directe : description simple
        return f"{prefix}Trajet {pickup_location} → {dropoff_location}"
