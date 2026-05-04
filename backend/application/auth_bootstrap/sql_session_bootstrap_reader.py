"""Lecture SQLAlchemy bornée pour le bootstrap de session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sqlalchemy.orm import joinedload

from application.auth_bootstrap.session_bootstrap_snapshot import (
    SessionBootstrapSnapshot,
)
from models import User
from models.enums import UserRole

if TYPE_CHECKING:
    pass


class SqlSessionBootstrapReader:
    """Charge `User` + relations strictement nécessaires en une requête."""

    def load_user_for_bootstrap(
        self, public_id: str
    ) -> tuple[SessionBootstrapSnapshot, User] | None:
        user = (
            User.query.options(
                joinedload(cast("Any", User.driver)),
                joinedload(cast("Any", User.clients)),
                joinedload(cast("Any", User.company)),
            )
            .filter_by(public_id=public_id)
            .first()
        )
        if user is None:
            return None

        drv = user.driver
        driver_id = drv.id if drv is not None else None
        driver_company_id = (
            cast(int | None, drv.company_id) if drv is not None else None
        )
        driver_is_active = bool(drv.is_active) if drv is not None else None

        company_rel = user.company
        company_relation_id = company_rel.id if company_rel is not None else None

        clients = list(user.clients) if user.clients is not None else []
        client_active_flags = tuple(bool(c.is_active) for c in clients)

        snap = SessionBootstrapSnapshot(
            user_id=cast(int, user.id),
            public_id=cast(str, user.public_id),
            username=cast(str, user.username),
            email=cast(str | None, user.email),
            role=cast(UserRole, user.role),
            account_status=cast(str | None, getattr(user, "account_status", None)),
            driver_id=driver_id,
            driver_company_id=driver_company_id,
            driver_is_active=driver_is_active,
            company_relation_id=company_relation_id,
            client_active_flags=client_active_flags,
        )
        return snap, user
