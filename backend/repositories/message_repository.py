"""Repository pour l'accès aux données Message."""

from datetime import datetime
from typing import Any, cast

from sqlalchemy.orm import joinedload

from models import Message

logger = __import__("logging").getLogger(__name__)


class MessageRepository:
    """Repository pour l'accès aux données Message."""

    def find_models_by_company_with_timestamp_filter(
        self, company_id: int, before_timestamp: datetime | None = None
    ) -> list[Message]:
        """Trouve les messages d'une entreprise avec filtre optionnel sur timestamp.

        Args:
            company_id: ID de l'entreprise
            before_timestamp: Timestamp maximum (optionnel)

        Returns:
            Liste de Message triés par timestamp descendant
        """
        query = Message.query.filter(Message.company_id == company_id).order_by(
            Message.timestamp.desc()
        )
        if before_timestamp:
            query = query.filter(Message.timestamp < before_timestamp)
        return query.all()

    def find_models_by_company_with_eager_loading(
        self,
        company_id: int,
        before_timestamp: datetime | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Trouve les messages d'une entreprise avec eager loading et filtres optionnels.

        Args:
            company_id: ID de l'entreprise
            before_timestamp: Timestamp maximum (optionnel)
            limit: Nombre maximum de résultats (optionnel)

        Returns:
            Liste de Message triés par timestamp descendant (avec sender et receiver chargés)
        """
        query = (
            Message.query.filter_by(company_id=company_id)
            .options(
                joinedload(cast("Any", Message.sender)),
                joinedload(cast("Any", Message.receiver)),
            )
            .order_by(Message.timestamp.desc())
        )
        if before_timestamp:
            query = query.filter(Message.timestamp < before_timestamp)
        if limit:
            query = query.limit(limit)
        return query.all()
