from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db
from models.enums import GeoUnitType


class GeoUnit(db.Model):
    __tablename__ = "geo_unit"
    __table_args__ = (
        UniqueConstraint("type", "code", name="uq_geo_unit_type_code"),
        Index("ix_geo_unit_type_parent", "type", "parent_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[GeoUnitType] = mapped_column(
        SAEnum(GeoUnitType, name="geo_unit_type", values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        nullable=False,
    )
    code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    parent_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("geo_unit.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    centroid_lat: Mapped[float | None] = mapped_column(nullable=True)
    centroid_lng: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    parent = relationship("GeoUnit", remote_side=[id], backref="children")

    def lineage(self) -> list[GeoUnit]:
        chain: list[GeoUnit] = []
        current: GeoUnit | None = self
        while current:
            chain.append(current)
            current = current.parent
        return chain
