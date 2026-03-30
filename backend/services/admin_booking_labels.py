"""Libellés FR des statuts réservation — source unique pour API admin (liste + détail)."""

from __future__ import annotations

from models.enums import BookingStatus

# Clés = valeur enum (ex. PENDING) ; affichage stable côté front via `status_label`.
STATUS_LABEL_FR: dict[str, str] = {
    BookingStatus.PENDING.value: "En attente",
    BookingStatus.ACCEPTED.value: "Acceptée",
    BookingStatus.ASSIGNED.value: "Assignée",
    BookingStatus.EN_ROUTE.value: "En route",
    BookingStatus.IN_PROGRESS.value: "En cours",
    BookingStatus.COMPLETED.value: "Terminée",
    BookingStatus.RETURN_COMPLETED.value: "Retour terminé",
    BookingStatus.CANCELED.value: "Annulée",
}


def booking_status_label_fr(status: BookingStatus | str | None) -> str:
    """Retourne le libellé français pour un statut booking."""
    if status is None:
        return "Inconnu"
    key = status.value if isinstance(status, BookingStatus) else str(status).upper()
    return STATUS_LABEL_FR.get(key, key.replace("_", " ").title())
