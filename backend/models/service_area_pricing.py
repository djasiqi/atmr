from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db
from models.enums import (
    DispatchOfferStatus,
    PricingModelType,
    ServiceCoverageMode,
)
from models.geo_unit import GeoUnit


class ServiceArea(db.Model):
    __tablename__ = "service_area"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "geo_unit_id",
            "coverage_mode",
            name="uq_service_area_company_geo_mode",
        ),
        Index("ix_service_area_company_active", "company_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    geo_unit_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("geo_unit.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    coverage_mode: Mapped[ServiceCoverageMode] = mapped_column(
        SAEnum(
            ServiceCoverageMode,
            name="service_coverage_mode",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
    )
    allow_pickup: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    allow_dropoff: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    weight: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
        default=0,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    company = relationship("Company", backref="service_areas")
    geo_unit = relationship("GeoUnit")


class PricingProfile(db.Model):
    __tablename__ = "pricing_profile"
    __table_args__ = (
        Index("ix_pricing_profile_company_active", "company_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    model_type: Mapped[PricingModelType] = mapped_column(
        SAEnum(
            PricingModelType,
            name="pricing_model_type",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
        server_default="CHF",
        default="CHF",
    )
    current_version_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("pricing_profile_version.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    company = relationship("Company", backref="pricing_profiles")
    current_version = relationship(
        "PricingProfileVersion",
        foreign_keys=[current_version_id],
        post_update=True,
    )


class PricingProfileVersion(db.Model):
    __tablename__ = "pricing_profile_version"
    __table_args__ = (
        UniqueConstraint(
            "pricing_profile_id",
            "version",
            name="uq_pricing_profile_version",
        ),
        Index("ix_pricing_profile_version_profile", "pricing_profile_id", "version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pricing_profile_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("pricing_profile.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    rules_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )

    pricing_profile = relationship(
        "PricingProfile",
        backref="versions",
        foreign_keys=[pricing_profile_id],
    )


class DispatchOffer(db.Model):
    __tablename__ = "dispatch_offer"
    __table_args__ = (
        UniqueConstraint(
            "booking_id", "company_id", name="uq_dispatch_offer_booking_company"
        ),
        Index("ix_dispatch_offer_booking_status", "booking_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[DispatchOfferStatus] = mapped_column(
        SAEnum(
            DispatchOfferStatus,
            name="dispatch_offer_status",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
    )
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    reason_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    booking = relationship("Booking", backref="dispatch_offers")
    company = relationship("Company", backref="dispatch_offers")


class PlatformZoneSet(db.Model):
    __tablename__ = "platform_zone_set"
    __table_args__ = (Index("ix_platform_zone_set_active_scope", "is_active", "scope"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    scope: Mapped[str | None] = mapped_column(String(16), nullable=True)
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1"), default=1
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PlatformZone(db.Model):
    __tablename__ = "platform_zone"
    __table_args__ = (
        UniqueConstraint("zone_set_id", "code", name="uq_platform_zone_set_code"),
        Index("ix_platform_zone_zone_set", "zone_set_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    zone_set_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_zone_set.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    code: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    zone_set = relationship("PlatformZoneSet", backref="zones")


class PlatformZoneMembership(db.Model):
    __tablename__ = "platform_zone_membership"
    __table_args__ = (
        UniqueConstraint(
            "zone_set_id",
            "commune_token",
            name="uq_platform_zone_membership_zone_set_commune",
        ),
        Index("ix_platform_zone_membership_zone_set", "zone_set_id"),
        Index("ix_platform_zone_membership_commune", "commune_token"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    zone_set_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_zone_set.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    zone_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_zone.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    commune_token: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    zone_set = relationship("PlatformZoneSet", backref="memberships")
    zone = relationship("PlatformZone", backref="memberships")


__all__ = [
    "DispatchOffer",
    "GeoUnit",
    "PlatformZone",
    "PlatformZoneMembership",
    "PlatformZoneSet",
    "PricingProfile",
    "PricingProfileVersion",
    "ServiceArea",
]
