# backend/models/trip_tracking_archive.py

"""✅ 3.5.2: Modèle pour table archive partitionnée des positions.

Table archive pour conserver l'historique des positions > 30 jours.
Partitionnée par mois pour performance et archivage facilité.
"""

import logging
from typing import Any, Dict

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, text
from typing_extensions import override

from ext import db

logger = logging.getLogger(__name__)


class TripTrackingArchive(db.Model):
    """Archive des positions pendant trajet (partitionnée par mois).

    Structure identique à TripTracking mais pour données archivées.
    Les partitions sont créées automatiquement par mois.
    """

    __tablename__ = "trip_tracking_archive"
    __table_args__ = (
        # Index sur les colonnes clés pour requêtes analytiques
        Index(
            "ix_trip_tracking_archive_assignment_timestamp",
            "assignment_id",
            "timestamp",
        ),
        Index("ix_trip_tracking_archive_booking_id", "booking_id"),
        Index("ix_trip_tracking_archive_driver_id", "driver_id"),
        Index("ix_trip_tracking_archive_timestamp", "timestamp"),
        # PostgreSQL partitioning sera géré via SQL direct
    )

    id = Column(Integer, primary_key=True)
    assignment_id = Column(
        Integer, ForeignKey("assignment.id"), nullable=False, index=True
    )
    booking_id = Column(Integer, ForeignKey("booking.id"), nullable=False, index=True)
    driver_id = Column(Integer, ForeignKey("driver.id"), nullable=False, index=True)

    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed = Column(Float, nullable=True)  # m/s
    heading = Column(Float, nullable=True)  # degrés (0-360)
    accuracy = Column(Float, nullable=True)  # mètres

    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "assignment_id": self.assignment_id,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "speed": self.speed,
            "heading": self.heading,
            "accuracy": self.accuracy,
            "timestamp": self.timestamp.isoformat() if bool(getattr(self, "timestamp", None)) else None,
        }

    @override
    def __repr__(self) -> str:
        return (
            f"<TripTrackingArchive id={self.id} assignment_id={self.assignment_id} "
            f"driver_id={self.driver_id} lat={self.latitude} lon={self.longitude}>"
        )

    @staticmethod
    def ensure_partition_for_month(year: int, month: int, db_session) -> bool:
        """Crée la partition pour un mois donné si elle n'existe pas.

        Args:
            year: Année (ex: 2025)
            month: Mois (1-12)
            db_session: Session SQLAlchemy

        Returns:
            True si partition créée, False si existait déjà
        """
        partition_name = f"trip_tracking_archive_{year}_{month:02d}"

        # Calculer les bornes de la partition
        from datetime import date

        DECEMBER_MONTH = 12
        start_date = date(year, month, 1)
        end_date = (
            date(year + 1, 1, 1)
            if month == DECEMBER_MONTH
            else date(year, month + 1, 1)
        )

        # Vérifier si partition existe déjà
        check_sql = text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_class
                WHERE relname = :partition_name
            )
        """)
        result = db_session.execute(
            check_sql, {"partition_name": partition_name}
        ).scalar()

        if result:
            logger.debug("Partition %s already exists", partition_name)
            return False

        # Créer la partition (utiliser format() pour nom de table, mais paramètres pour dates)
        # Note: Le nom de la partition doit être dans le SQL, pas un paramètre
        create_sql = text(
            f"""
            CREATE TABLE IF NOT EXISTS {partition_name}
            PARTITION OF trip_tracking_archive
            FOR VALUES FROM (:start_date) TO (:end_date)
        """
        )

        try:
            db_session.execute(
                create_sql,
                {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )
            db_session.commit()
            logger.info("Created partition %s for %d-%02d", partition_name, year, month)
            return True
        except Exception as e:
            logger.error("Failed to create partition %s: %s", partition_name, e)
            db_session.rollback()
            raise
