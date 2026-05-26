"""Central permission checks for messaging (HTTP + socket)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from models import Conversation, ConversationParticipant, Driver, Message, User
from models.messaging_enums import (
    ConversationType,
    DEFAULT_MESSAGE_VISIBILITY_TAGS,
    ParticipantRole,
)

if TYPE_CHECKING:
    from models.enums import UserRole


def _normalized_user_role(user: User) -> str:
    """Rôle utilisateur en majuscules (``UserRole.COMPANY`` → ``COMPANY``)."""
    return str(getattr(user.role, "value", user.role)).upper()


class MessagingPermissionService:
    """Single source of truth for messaging authorization."""

    @staticmethod
    def participant_for(
        conversation_id: int, user_id: int
    ) -> ConversationParticipant | None:
        return ConversationParticipant.query.filter_by(
            conversation_id=conversation_id,
            user_id=user_id,
            left_at=None,
        ).first()

    @staticmethod
    def can_read_conversation(user: User, conversation: Conversation) -> bool:
        role = _normalized_user_role(user)
        if role == "COMPANY":
            return int(conversation.company_id) == _company_id_for_user(user)
        if role == "DRIVER":
            driver = getattr(user, "driver", None)
            if not driver or int(driver.company_id) != int(conversation.company_id):
                return False
            part = MessagingPermissionService.participant_for(conversation.id, user.id)
            return part is not None and bool(part.can_read)
        return False

    @staticmethod
    def can_write_conversation(user: User, conversation: Conversation) -> bool:
        if not MessagingPermissionService.can_read_conversation(user, conversation):
            return False
        role = _normalized_user_role(user)
        if role == "COMPANY":
            return True
        part = MessagingPermissionService.participant_for(conversation.id, user.id)
        return part is not None and bool(part.can_write)

    @staticmethod
    def can_manage_conversation(user: User, conversation: Conversation) -> bool:
        role = _normalized_user_role(user)
        if role != "COMPANY":
            return False
        if int(conversation.company_id) != _company_id_for_user(user):
            return False
        return True

    @staticmethod
    def can_create_group(user: User) -> bool:
        return _normalized_user_role(user) == "COMPANY"

    @staticmethod
    def can_create_direct(user: User) -> bool:
        """Chauffeurs de la même entreprise peuvent démarrer un DM collègue."""
        return (
            _normalized_user_role(user) == "DRIVER"
            and getattr(user, "driver", None) is not None
        )

    @staticmethod
    def can_direct_message_peer(user: User, peer_user_id: int) -> bool:
        if not MessagingPermissionService.can_create_direct(user):
            return False
        driver = user.driver
        if int(user.id) == int(peer_user_id):
            return False
        peer = Driver.query.filter_by(user_id=int(peer_user_id)).first()
        if not peer or not peer.is_active:
            return False
        return int(peer.company_id) == int(driver.company_id)

    @staticmethod
    def assert_can_direct_message_peer(user: User, peer_user_id: int) -> None:
        if not MessagingPermissionService.can_direct_message_peer(user, peer_user_id):
            raise PermissionError("Message direct refusé pour ce collègue")

    @staticmethod
    def can_read_message(
        user: User,
        message: Message,
        conversation: Conversation | None,
        participant: ConversationParticipant | None,
    ) -> bool:
        if conversation is None:
            return False
        if not participant or not participant.can_read:
            if not MessagingPermissionService.can_read_conversation(user, conversation):
                return False
        # V1: visibility_scope is all_participants — no tag filtering yet
        _ = conversation.visibility_scope
        tags = message.visibility_tags or DEFAULT_MESSAGE_VISIBILITY_TAGS
        if not tags:
            return True
        return True

    @staticmethod
    def assert_can_read(user: User, conversation: Conversation) -> None:
        if not MessagingPermissionService.can_read_conversation(user, conversation):
            raise PermissionError("Accès conversation refusé")

    @staticmethod
    def assert_can_manage(user: User, conversation: Conversation) -> None:
        if not MessagingPermissionService.can_manage_conversation(user, conversation):
            raise PermissionError("Gestion conversation refusée")

    @staticmethod
    def assert_can_write(user: User, conversation: Conversation) -> None:
        if not MessagingPermissionService.can_write_conversation(user, conversation):
            raise PermissionError("Écriture conversation refusée")


def _company_id_for_user(user: User) -> int:
    company = getattr(user, "company", None)
    if company is not None:
        return int(company.id)
    raise PermissionError("Entreprise introuvable")


def driver_assigned_to_booking(driver: Driver, booking_id: int) -> bool:
    from models import Booking

    booking = Booking.query.filter_by(
        id=booking_id, company_id=driver.company_id
    ).first()
    if not booking:
        return False
    return int(getattr(booking, "driver_id", 0) or 0) == int(driver.id)
