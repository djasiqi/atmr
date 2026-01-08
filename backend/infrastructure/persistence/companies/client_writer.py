from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ext import db
from models import Client, ClientType, User, UserRole


@dataclass(frozen=True, slots=True)
class SqlAlchemyClientWriter:
    """Adaptateur Infrastructure: création d'un client (User + Client)
    via SQLAlchemy."""

    def create_client_for_company(
        self,
        *,
        company_id: int,
        user_attrs: dict[str, Any],
        client_attrs: dict[str, Any],
    ) -> tuple[User, Client]:
        user = User()
        user.public_id = user_attrs.get("public_id")
        user.username = user_attrs.get("username")
        user.first_name = user_attrs.get("first_name") or ""
        user.last_name = user_attrs.get("last_name") or ""
        user.email = user_attrs.get("email")
        user.phone = user_attrs.get("phone")
        user.address = user_attrs.get("address")
        user.birth_date = user_attrs.get("birth_date")
        user.role = UserRole.client

        password = user_attrs.get("password")
        if password:
            user.set_password(password)  # nosem

        db.session.add(user)
        db.session.flush()

        ct = client_attrs.get("client_type") or "PRIVATE"
        ct_upper = str(ct).upper()
        client_type = (
            ClientType[ct_upper]
            if ct_upper in ClientType.__members__
            else ClientType.PRIVATE
        )

        preferential_rate = client_attrs.get("preferential_rate")
        if preferential_rate not in (None, ""):
            try:
                preferential_rate = Decimal(str(preferential_rate))
            except Exception:
                preferential_rate = None
        else:
            preferential_rate = None

        client = Client()
        client.user_id = user.id
        client.company_id = company_id
        client.client_type = client_type
        client.billing_address = client_attrs.get("billing_address")
        client.billing_lat = client_attrs.get("billing_lat")
        client.billing_lon = client_attrs.get("billing_lon")
        client.contact_email = client_attrs.get("contact_email")
        client.contact_phone = client_attrs.get("contact_phone")
        client.is_institution = bool(client_attrs.get("is_institution", False))
        client.institution_name = client_attrs.get("institution_name")
        client.domicile_address = client_attrs.get("domicile_address")
        client.domicile_zip = client_attrs.get("domicile_zip")
        client.domicile_city = client_attrs.get("domicile_city")
        client.domicile_lat = client_attrs.get("domicile_lat")
        client.domicile_lon = client_attrs.get("domicile_lon")
        client.preferential_rate = preferential_rate
        client.residence_facility = client_attrs.get("residence_facility")

        db.session.add(client)
        return user, client
