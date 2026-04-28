"""Résolution de l'entreprise « propriétaire » lors de la création d'une réservation."""

from __future__ import annotations

from typing import Any

from models.enums import ClientType


def resolve_booking_owner_company_id_for_create(client_dto: Any) -> int | None:
    """Détermine le `company_id` à enregistrer sur le booking à la création.

    Règles strictes par ``ClientType`` :

    - ``PORTAL`` → ``None`` (marché ouvert, pas d'entreprise pré-assignée).
    - ``TRANSPORT`` → ``company_id`` obligatoire.
      Un client TRANSPORT sans ``company_id`` est un état invalide.

    Returns:
        ID entreprise propriétaire, ou ``None`` pour le marché ouvert.

    Raises:
        ValueError: client TRANSPORT sans company_id ou type inconnu.
    """
    ct = getattr(client_dto, "client_type", None)

    if ct == ClientType.PORTAL:
        return None

    if ct == ClientType.TRANSPORT:
        cid = int(getattr(client_dto, "company_id", None) or 0)
        if cid > 0:
            return cid
        msg = (
            "Client TRANSPORT sans company_id. "
            "État invalide : un client TRANSPORT doit toujours être "
            "rattaché à une entreprise de transport."
        )
        raise ValueError(msg)

    raise ValueError(f"ClientType non géré : {ct}")
