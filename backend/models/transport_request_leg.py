"""Segments ordonnés d'une demande multi-étapes (TransportRequest).



Invariant horaire :

- ``time_confirmed`` est persisté en base (indépendant de ``scheduled_time``).

- ``scheduled_time`` peut exister sans confirmation (heure indicative).

- ``time_confirmed=true`` implique toujours ``scheduled_time != null``.

- Heure indicative : affichage autorisé, exclue des métriques opérationnelles

  (voir ``mission_schedule.is_operational_time``).

"""



from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from ext import db


def _iso_scheduled(dt):
    """ISO naïf Genève pour horaires mission (contrat API institution)."""
    if dt is None:
        return None
    from shared.time_utils import mission_scheduled_to_api_iso

    return mission_scheduled_to_api_iso(dt)



if TYPE_CHECKING:

    from .transport_request import TransportRequest





class TransportRequestLeg(db.Model):

    __tablename__ = "transport_request_legs"

    __table_args__ = (

        Index(

            "uq_transport_request_leg_sequence",

            "transport_request_id",

            "sequence_index",

            unique=True,

        ),

    )



    id: Mapped[int] = mapped_column(Integer, primary_key=True)



    transport_request_id: Mapped[int] = mapped_column(

        ForeignKey("transport_requests.id", ondelete="CASCADE"),

        nullable=False,

        index=True,

    )



    sequence_index: Mapped[int] = mapped_column(Integer, nullable=False)

    route_sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)



    pickup_location: Mapped[str] = mapped_column(String(255), nullable=False)

    pickup_lat: Mapped[Decimal | None] = mapped_column(Numeric(10, 7), nullable=True)

    pickup_lng: Mapped[Decimal | None] = mapped_column(Numeric(10, 7), nullable=True)

    dropoff_location: Mapped[str] = mapped_column(String(255), nullable=False)

    dropoff_lat: Mapped[Decimal | None] = mapped_column(Numeric(10, 7), nullable=True)

    dropoff_lng: Mapped[Decimal | None] = mapped_column(Numeric(10, 7), nullable=True)



    dropoff_establishment: Mapped[str | None] = mapped_column(

        String(255), nullable=True

    )

    dropoff_service: Mapped[str | None] = mapped_column(String(255), nullable=True)

    dropoff_doctor: Mapped[str | None] = mapped_column(String(255), nullable=True)



    scheduled_time = mapped_column(DateTime(timezone=False), nullable=True)

    time_confirmed: Mapped[bool] = mapped_column(

        Boolean,

        nullable=False,

        default=False,

        server_default="false",

    )



    booking_id: Mapped[int | None] = mapped_column(

        ForeignKey("booking.id", ondelete="SET NULL"),

        nullable=True,

    )



    created_at = mapped_column(

        DateTime(timezone=True), server_default=func.now(), nullable=False

    )



    transport_request: Mapped[TransportRequest] = relationship(

        "TransportRequest",

        back_populates="legs",

    )



    @validates("time_confirmed")

    def validate_time_confirmed(self, _key: str, value: bool) -> bool:

        if value and self.scheduled_time is None:

            raise ValueError(

                "time_confirmed=true requiert scheduled_time renseigné sur le leg."

            )

        return value



    def serialize(self) -> dict[str, Any]:

        return {

            "id": self.id,

            "transport_request_id": self.transport_request_id,

            "sequence_index": self.sequence_index,

            "route_sequence_number": self.route_sequence_number,

            "pickup_location": self.pickup_location,

            "pickup_lat": float(self.pickup_lat) if self.pickup_lat else None,

            "pickup_lng": float(self.pickup_lng) if self.pickup_lng else None,

            "dropoff_location": self.dropoff_location,

            "dropoff_lat": float(self.dropoff_lat) if self.dropoff_lat else None,

            "dropoff_lng": float(self.dropoff_lng) if self.dropoff_lng else None,

            "dropoff_establishment": self.dropoff_establishment,

            "dropoff_service": self.dropoff_service,

            "dropoff_doctor": self.dropoff_doctor,

            "scheduled_time": _iso_scheduled(self.scheduled_time),

            "time_confirmed": bool(self.time_confirmed),

            "booking_id": self.booking_id,

        }


