"""Interface abstraite pour le service de notifications.

Cette interface permet de :
1. Faciliter les tests (mocks)
2. Préparer une migration vers microservices (remplacement par API REST)
3. Améliorer la séparation des responsabilités
"""
# pyright: reportImplicitOverride=false
# Note: Les méthodes de NotificationServiceLocal utilisent @override mais basedpyright
# ne le reconnaît pas toujours dans ce contexte (problème connu avec les imports conditionnels)

from abc import ABC, abstractmethod
from typing import Any, Dict

# Import override - typing_extensions est garanti disponible (dans requirements.base.txt)
try:
    from typing import (
        override,
    )
except ImportError:
    from typing_extensions import override  # Python < 3.12


class NotificationServiceInterface(ABC):
    """Interface pour le service de notifications."""

    @abstractmethod
    def send_push_message(
        self, token: str, title: str, body: str, *, timeout: int = 5
    ) -> Dict[str, Any]:
        """Envoie un message push via Expo.

        Args:
            token: Token Expo du destinataire
            title: Titre du message
            body: Corps du message
            timeout: Timeout en secondes

        Returns:
            Dict avec résultat de l'envoi
        """
        pass

    @abstractmethod
    def notify_booking_assigned(self, booking: Any, driver: Any | None = None) -> None:
        """Notifie qu'une réservation a été assignée.

        Args:
            booking: Objet Booking
            driver: Objet Driver (optionnel, peut être extrait de booking)
        """
        pass

    @abstractmethod
    def notify_dispatch_run_completed(
        self, company_id: int, run_id: int, result: Dict[str, Any]
    ) -> None:
        """Notifie qu'un run de dispatch est terminé.

        Args:
            company_id: ID de l'entreprise
            run_id: ID du run
            result: Résultat du dispatch
        """
        pass

    @abstractmethod
    def emit_company_event(
        self, company_id: int, event: str, payload: Dict[str, Any]
    ) -> None:
        """Émet un événement Socket.IO pour une entreprise.

        Args:
            company_id: ID de l'entreprise
            event: Nom de l'événement
            payload: Données de l'événement
        """
        pass


class NotificationServiceLocal(NotificationServiceInterface):
    """Implémentation locale (monolithique) du service de notifications."""

    @override
    def send_push_message(
        self, token: str, title: str, body: str, *, timeout: int = 5
    ) -> Dict[str, Any]:
        """Implémentation locale via services.notification_service."""
        from services.notification_service import (
            send_push_message as _send_push,
        )

        return _send_push(token, title, body, timeout=timeout)

    @override
    def notify_booking_assigned(
        self,
        booking: Any,
        driver: Any | None = None,  # driver non utilisé mais requis par l'interface
    ) -> None:
        """Implémentation locale via services.notification_service."""
        from services.notification_service import (
            notify_booking_assigned as _notify_assigned,
        )

        # notify_booking_assigned n'accepte qu'un seul argument (booking)
        _notify_assigned(booking)

    @override
    def notify_dispatch_run_completed(
        self, company_id: int, run_id: int, result: Dict[str, Any]
    ) -> None:
        """Implémentation locale via services.notification_service."""
        from services.notification_service import (
            notify_dispatch_run_completed as _notify_completed,
        )

        # Extraire les compteurs du résultat
        assignments_count = len(result.get("assignments", []))
        _notify_completed(company_id, run_id, assignments_count)

    @override
    def emit_company_event(
        self, company_id: int, event: str, payload: Dict[str, Any]
    ) -> None:
        """Implémentation locale via services.realtime.socketio."""
        from services.realtime.socketio import emit_company_event as _emit_company

        _emit_company(company_id, event, payload)


# Instance par défaut (monolithique)
_default_notification_service: NotificationServiceInterface = NotificationServiceLocal()


def get_notification_service() -> NotificationServiceInterface:
    """Récupère l'instance du service de notifications.

    Dans une architecture microservices, cette fonction pourrait retourner
    un client HTTP vers le service de notifications distant.

    Returns:
        Instance du service de notifications
    """
    return _default_notification_service


def set_notification_service(service: NotificationServiceInterface) -> None:
    """Définit l'instance du service de notifications (pour tests).

    Args:
        service: Instance du service de notifications
    """
    # Mettre à jour via le module pour éviter global statement
    import services.interfaces.notification_interface as module

    module._default_notification_service = service

