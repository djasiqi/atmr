"""Implémentation SQLAlchemy du repository Driver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from drivers.domain.driver import Driver
from drivers.domain.driver_id import DriverId

if TYPE_CHECKING:
    from models import Driver as SQLAlchemyDriver
else:
    SQLAlchemyDriver = Any

logger = __import__("logging").getLogger(__name__)


class SqlAlchemyDriverRepository:
    """Implémentation SQLAlchemy du repository Driver.

    Adapte les modèles SQLAlchemy vers les agrégats du domaine.
    """

    def _to_aggregate(self, sa_driver: SQLAlchemyDriver) -> Driver:
        """Convertit un modèle SQLAlchemy en agrégat Driver."""
        from drivers.domain.value_objects import (
            DriverLocation,
            DriverStatus,
            DriverType,
        )

        # Construire DriverLocation si les coordonnées sont présentes
        location = None
        if (
            getattr(sa_driver, "latitude", None) is not None
            and getattr(sa_driver, "longitude", None) is not None
        ):
            location = DriverLocation(
                latitude=float(getattr(sa_driver, "latitude", 0.0)),
                longitude=float(getattr(sa_driver, "longitude", 0.0)),
                # Par défaut, peut être enrichi depuis DriverStatus si dispo
                accuracy=0.0,
                timestamp=sa_driver.last_position_update
                or __import__("datetime").datetime.now(),
                speed=None,  # Peut être enrichi depuis DriverStatus
                heading=None,  # Peut être enrichi depuis DriverStatus
            )

        # Construire DriverStatus
        driver_type = DriverType(str(sa_driver.driver_type.value))
        status = DriverStatus(
            is_active=bool(sa_driver.is_active),
            is_available=bool(sa_driver.is_available),
            driver_type=driver_type,
        )

        return Driver(
            id=DriverId(sa_driver.id),
            user_id=cast(int, sa_driver.user_id),
            company_id=cast(int, sa_driver.company_id),
            status=status,
            location=location,
            vehicle_assigned=sa_driver.vehicle_assigned,
            brand=sa_driver.brand,
            license_plate=sa_driver.license_plate,
            push_token=sa_driver.push_token,
            created_at=cast(
                Any | None,
                sa_driver.created_at if hasattr(sa_driver, "created_at") else None,
            ),
            updated_at=cast(
                Any | None,
                sa_driver.updated_at if hasattr(sa_driver, "updated_at") else None,
            ),
        )

    def _from_aggregate(self, driver: Driver) -> dict[str, Any]:
        """Convertit un agrégat Driver en dictionnaire pour SQLAlchemy."""
        data: dict[str, Any] = {
            "id": driver.id.value,
            "user_id": driver.user_id,
            "company_id": driver.company_id,
            "is_active": driver.status.is_active,
            "is_available": driver.status.is_available,
            "driver_type": driver.status.driver_type.value,
            "vehicle_assigned": driver.vehicle_assigned,
            "brand": driver.brand,
            "license_plate": driver.license_plate,
            "push_token": driver.push_token,
        }

        # Ajouter les coordonnées si location est présente
        if driver.location:
            data["latitude"] = driver.location.latitude
            data["longitude"] = driver.location.longitude
            data["last_position_update"] = driver.location.timestamp
        else:
            data["latitude"] = None
            data["longitude"] = None
            data["last_position_update"] = None

        return data

    def save(self, driver: Driver) -> None:
        """Sauvegarde un chauffeur."""
        from ext import db
        from models import Driver as SQLAlchemyDriver

        data = self._from_aggregate(driver)
        driver_id = data.pop("id")

        sa_driver = SQLAlchemyDriver.query.get(driver_id)
        if sa_driver:
            # Update
            for key, value in data.items():
                setattr(sa_driver, key, value)
        else:
            # Create
            sa_driver = SQLAlchemyDriver(**data)
            db.session.add(sa_driver)

        db.session.commit()

    def find_by_id(self, driver_id: DriverId) -> Driver | None:
        """Trouve un chauffeur par ID."""
        from models import Driver as SQLAlchemyDriver

        sa_driver = SQLAlchemyDriver.query.get(driver_id.value)
        if sa_driver is None:
            return None
        return self._to_aggregate(sa_driver)

    def find_by_company_id(self, company_id: int) -> list[Driver]:
        """Trouve tous les chauffeurs d'une entreprise."""
        from models import Driver as SQLAlchemyDriver

        sa_drivers = SQLAlchemyDriver.query.filter_by(company_id=company_id).all()
        return [self._to_aggregate(d) for d in sa_drivers]

    def find_available_by_company(self, company_id: int) -> list[Driver]:
        """Trouve tous les chauffeurs disponibles d'une entreprise."""
        from models import Driver as SQLAlchemyDriver

        sa_drivers = SQLAlchemyDriver.query.filter_by(
            company_id=company_id, is_active=True, is_available=True
        ).all()
        return [self._to_aggregate(d) for d in sa_drivers]

    def find_by_user_id(self, user_id: int) -> Driver | None:
        """Trouve un chauffeur par user_id."""
        from models import Driver as SQLAlchemyDriver

        sa_driver = SQLAlchemyDriver.query.filter_by(user_id=user_id).first()
        if sa_driver is None:
            return None
        return self._to_aggregate(sa_driver)
