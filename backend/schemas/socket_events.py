# schemas/socket_events.py
"""Schémas centralisés pour les événements Socket.IO.

Ce module définit les contrats (schemas) pour tous les événements Socket.IO
émis par le système, garantissant la cohérence et la maintenabilité.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict


@dataclass
class SocketEvent:
    """Contrat centralisé pour un événement Socket.IO.

    Tous les événements Socket.IO doivent respecter ce schéma pour garantir :
    - Cohérence des métadonnées (event_id, version, timestamp)
    - Traçabilité et déduplication
    - Versioning futur des événements

    Attributes:
        event_id: UUID v4 unique pour cet événement (déduplication)
        version: Version du format d'événement (ex: "1.0")
        timestamp: Timestamp ISO 8601 UTC de l'émission
        event_type: Type d'événement métier (ex: "booking_assigned", "dispatch_completed")
        payload: Données métier de l'événement (fusionnées avec les métadonnées)
    """

    event_id: str
    version: str
    timestamp: str
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: str,
        payload: Dict[str, Any],
        version: str = "1.0",
    ) -> Dict[str, Any]:
        """Crée un payload d'événement Socket.IO enrichi avec métadonnées.

        Cette méthode génère automatiquement les métadonnées requises et fusionne
        le payload métier avec ces métadonnées.

        Args:
            event_type: Type d'événement métier (ex: "booking_assigned", "team_chat_message")
            payload: Données métier de l'événement (seront fusionnées avec les métadonnées)
            version: Version du format d'événement (défaut: "1.0")

        Returns:
            Dict enrichi contenant:
            - event_id: UUID v4 unique
            - version: Version du format
            - timestamp: Timestamp ISO 8601 UTC
            - event_type: Type d'événement métier
            - ...tous les champs du payload (fusionnés)

        Example:
            ```python
            payload = SocketEvent.create(
                event_type="booking_assigned",
                payload={"id": 123, "driver_id": 456}
            )
            # Résultat:
            # {
            #     "event_id": "550e8400-e29b-41d4-a716-446655440000",
            #     "version": "1.0",
            #     "timestamp": "2025-01-20T10:30:00.000Z",
            #     "event_type": "booking_assigned",
            #     "id": 123,
            #     "driver_id": 456
            # }
            ```
        """
        event = cls(
            event_id=str(uuid.uuid4()),
            version=version,
            timestamp=datetime.now(UTC).isoformat(),
            event_type=event_type,
            payload=payload,
        )

        # Fusionner les métadonnées avec le payload métier
        return {
            "event_id": event.event_id,
            "version": event.version,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            **event.payload,
        }

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> bool:
        """Valide qu'un dict respecte le schéma SocketEvent.

        Args:
            data: Dict à valider

        Returns:
            True si le dict contient les champs requis (event_id, version, timestamp, event_type)
        """
        required_fields = {"event_id", "version", "timestamp", "event_type"}
        return all(field in data for field in required_fields)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'instance SocketEvent en dict.

        Returns:
            Dict contenant toutes les métadonnées fusionnées avec le payload
        """
        return {
            "event_id": self.event_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            **self.payload,
        }


# Version globale des événements (pour versioning futur)
EVENT_VERSION = "1.0"
