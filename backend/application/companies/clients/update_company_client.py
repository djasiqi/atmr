from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol


class _UserLike(Protocol):
    birth_date: Any


class _ClientLike(Protocol):
    user: _UserLike | None
    contact_email: Any
    contact_phone: Any
    billing_address: Any
    billing_lat: Any
    billing_lon: Any
    is_active: Any
    is_institution: Any
    institution_name: Any
    residence_facility: Any
    domicile_address: Any
    domicile_zip: Any
    domicile_city: Any
    domicile_lat: Any
    domicile_lon: Any
    preferential_rate: Any


@dataclass(frozen=True, slots=True)
class UpdateCompanyClientResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None


class UpdateCompanyClientUseCase:
    """Use-case Application: mise à jour d'un client
    (coordonnées, facturation, statut, etc.)."""

    def execute(
        self, *, client: _ClientLike, data: dict[str, Any]
    ) -> UpdateCompanyClientResult:
        # Champs Client
        if "contact_email" in data:
            client.contact_email = data["contact_email"]
        if "contact_phone" in data:
            client.contact_phone = data["contact_phone"]
        if "billing_address" in data:
            client.billing_address = data["billing_address"]
        if "billing_lat" in data:
            client.billing_lat = data["billing_lat"]
        if "billing_lon" in data:
            client.billing_lon = data["billing_lon"]
        if "is_active" in data:
            client.is_active = bool(data["is_active"])

        if "is_institution" in data:
            client.is_institution = bool(data["is_institution"])
            if bool(client.is_institution) and "institution_name" in data:
                client.institution_name = data["institution_name"]
            elif not bool(client.is_institution):
                client.institution_name = None

        if "residence_facility" in data:
            client.residence_facility = data["residence_facility"] or None

        if "domicile_address" in data:
            client.domicile_address = data["domicile_address"] or None
        if "domicile_zip" in data:
            client.domicile_zip = data["domicile_zip"] or None
        if "domicile_city" in data:
            client.domicile_city = data["domicile_city"] or None
        if "domicile_lat" in data:
            client.domicile_lat = data["domicile_lat"]
        if "domicile_lon" in data:
            client.domicile_lon = data["domicile_lon"]

        if "preferential_rate" in data:
            rate_value = data["preferential_rate"]
            if rate_value == "" or rate_value is None:
                client.preferential_rate = None
            else:
                try:
                    client.preferential_rate = Decimal(str(rate_value))
                except (ValueError, TypeError):
                    return UpdateCompanyClientResult(
                        ok=False,
                        error={"error": "Tarif préférentiel invalide"},
                        status_code=400,
                    )

        # Champ User.birth_date (format YYYY-MM-DD)
        if "birth_date" in data and getattr(client, "user", None):
            user = client.user
            assert user is not None
            birth_date_value = data["birth_date"]
            if birth_date_value:
                try:
                    user.birth_date = datetime.strptime(
                        str(birth_date_value), "%Y-%m-%d"
                    ).date()
                except (ValueError, TypeError):
                    return UpdateCompanyClientResult(
                        ok=False,
                        error={
                            "error": (
                                "Format de date de naissance invalide. "
                                "Utiliser YYYY-MM-DD."
                            )
                        },
                        status_code=400,
                    )
            else:
                user.birth_date = None

        return UpdateCompanyClientResult(ok=True)
