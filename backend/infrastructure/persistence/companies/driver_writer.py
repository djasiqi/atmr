from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ext import db
from models import Driver, User, UserRole


@dataclass(frozen=True, slots=True)
class SqlAlchemyDriverWriter:
    """Adaptateur Infrastructure: création d'un chauffeur (User + Driver) via SQLAlchemy."""

    def create_driver_for_company(
        self,
        *,
        company_id: int,
        user_attrs: dict[str, Any],
        driver_attrs: dict[str, Any],
    ) -> tuple[User, Driver]:
        new_user = User()
        new_user.username = user_attrs["username"]
        new_user.first_name = user_attrs.get("first_name")
        new_user.last_name = user_attrs.get("last_name")
        new_user.email = user_attrs.get("email")
        new_user.role = UserRole.driver
        new_user.public_id = user_attrs.get("public_id")

        password = user_attrs.get("password")
        if password:
            new_user.set_password(password)  # nosem

        db.session.add(new_user)
        db.session.flush()

        new_driver = Driver()
        new_driver.user_id = new_user.id
        new_driver.company_id = company_id
        new_driver.vehicle_assigned = driver_attrs.get("vehicle_assigned")
        new_driver.brand = driver_attrs.get("brand")
        new_driver.license_plate = driver_attrs.get("license_plate")
        new_driver.is_active = bool(driver_attrs.get("is_active", True))
        new_driver.is_available = bool(driver_attrs.get("is_available", True))

        db.session.add(new_driver)
        return new_user, new_driver
