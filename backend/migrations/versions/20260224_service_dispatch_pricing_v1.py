"""Add service areas, geo units, pricing profiles and dispatch offers.

Revision ID: 20260224_service_dispatch
Revises: 20260222_merge
Create Date: 2026-02-24 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260224_service_dispatch"
down_revision = "20260222_merge"
branch_labels = None
depends_on = None


geo_unit_type = postgresql.ENUM(
    "country",
    "canton",
    "district",
    "commune",
    "zipcode",
    name="geo_unit_type",
    create_type=False,
)
service_coverage_mode = postgresql.ENUM(
    "A_STRICT",
    "B_PICKUP_ONLY",
    "C_INTRA_ONLY",
    "D_NATIONAL",
    name="service_coverage_mode",
    create_type=False,
)
pricing_model_type = postgresql.ENUM(
    "flat", "zone", "distance", "hybrid", name="pricing_model_type", create_type=False
)
dispatch_offer_status = postgresql.ENUM(
    "PROPOSED",
    "ACCEPTED",
    "DECLINED",
    "EXPIRED",
    name="dispatch_offer_status",
    create_type=False,
)


def upgrade():
    bind = op.get_bind()
    geo_unit_type.create(bind, checkfirst=True)
    service_coverage_mode.create(bind, checkfirst=True)
    pricing_model_type.create(bind, checkfirst=True)
    dispatch_offer_status.create(bind, checkfirst=True)

    op.create_table(
        "geo_unit",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("type", geo_unit_type, nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("centroid_lat", sa.Float(), nullable=True),
        sa.Column("centroid_lng", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["parent_id"], ["geo_unit.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("type", "code", name="uq_geo_unit_type_code"),
    )
    op.create_index(op.f("ix_geo_unit_code"), "geo_unit", ["code"], unique=False)
    op.create_index(op.f("ix_geo_unit_parent_id"), "geo_unit", ["parent_id"], unique=False)
    op.create_index("ix_geo_unit_type_parent", "geo_unit", ["type", "parent_id"], unique=False)

    op.create_table(
        "service_area",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("geo_unit_id", sa.Integer(), nullable=False),
        sa.Column("coverage_mode", service_coverage_mode, nullable=False),
        sa.Column(
            "allow_pickup",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "allow_dropoff",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column("weight", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["geo_unit_id"], ["geo_unit.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "company_id",
            "geo_unit_id",
            "coverage_mode",
            name="uq_service_area_company_geo_mode",
        ),
    )
    op.create_index(op.f("ix_service_area_company_id"), "service_area", ["company_id"], unique=False)
    op.create_index(op.f("ix_service_area_geo_unit_id"), "service_area", ["geo_unit_id"], unique=False)
    op.create_index("ix_service_area_company_active", "service_area", ["company_id", "is_active"], unique=False)

    op.create_table(
        "pricing_profile",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("model_type", pricing_model_type, nullable=False),
        sa.Column("currency", sa.String(length=3), server_default="CHF", nullable=False),
        sa.Column("current_version_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_pricing_profile_company_id"), "pricing_profile", ["company_id"], unique=False)
    op.create_index(
        "ix_pricing_profile_company_active",
        "pricing_profile",
        ["company_id", "is_active"],
        unique=False,
    )

    op.create_table(
        "pricing_profile_version",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pricing_profile_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("rules_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["pricing_profile_id"], ["pricing_profile.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "pricing_profile_id",
            "version",
            name="uq_pricing_profile_version",
        ),
    )
    op.create_index(
        op.f("ix_pricing_profile_version_pricing_profile_id"),
        "pricing_profile_version",
        ["pricing_profile_id"],
        unique=False,
    )
    op.create_index(
        "ix_pricing_profile_version_profile",
        "pricing_profile_version",
        ["pricing_profile_id", "version"],
        unique=False,
    )

    op.create_foreign_key(
        "fk_pricing_profile_current_version",
        "pricing_profile",
        "pricing_profile_version",
        ["current_version_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_table(
        "dispatch_offer",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("status", dispatch_offer_status, nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("reason_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("booking_id", "company_id", name="uq_dispatch_offer_booking_company"),
    )
    op.create_index(op.f("ix_dispatch_offer_booking_id"), "dispatch_offer", ["booking_id"], unique=False)
    op.create_index(op.f("ix_dispatch_offer_company_id"), "dispatch_offer", ["company_id"], unique=False)
    op.create_index("ix_dispatch_offer_booking_status", "dispatch_offer", ["booking_id", "status"], unique=False)

    op.add_column("booking", sa.Column("pickup_geo_unit_id", sa.Integer(), nullable=True))
    op.add_column("booking", sa.Column("dropoff_geo_unit_id", sa.Integer(), nullable=True))
    op.add_column("booking", sa.Column("pickup_zip", sa.String(length=16), nullable=True))
    op.add_column("booking", sa.Column("dropoff_zip", sa.String(length=16), nullable=True))
    op.add_column("booking", sa.Column("pricing_profile_id", sa.Integer(), nullable=True))
    op.add_column("booking", sa.Column("pricing_profile_version_id", sa.Integer(), nullable=True))
    op.add_column("booking", sa.Column("price_amount", sa.Numeric(precision=10, scale=2), nullable=True))
    op.add_column(
        "booking",
        sa.Column("price_breakdown_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(op.f("ix_booking_pickup_geo_unit_id"), "booking", ["pickup_geo_unit_id"], unique=False)
    op.create_index(op.f("ix_booking_dropoff_geo_unit_id"), "booking", ["dropoff_geo_unit_id"], unique=False)
    op.create_index(op.f("ix_booking_pricing_profile_id"), "booking", ["pricing_profile_id"], unique=False)
    op.create_index(
        op.f("ix_booking_pricing_profile_version_id"),
        "booking",
        ["pricing_profile_version_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_booking_pickup_geo_unit",
        "booking",
        "geo_unit",
        ["pickup_geo_unit_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_booking_dropoff_geo_unit",
        "booking",
        "geo_unit",
        ["dropoff_geo_unit_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_booking_pricing_profile",
        "booking",
        "pricing_profile",
        ["pricing_profile_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_booking_pricing_profile_version",
        "booking",
        "pricing_profile_version",
        ["pricing_profile_version_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    op.drop_constraint("fk_booking_pricing_profile_version", "booking", type_="foreignkey")
    op.drop_constraint("fk_booking_pricing_profile", "booking", type_="foreignkey")
    op.drop_constraint("fk_booking_dropoff_geo_unit", "booking", type_="foreignkey")
    op.drop_constraint("fk_booking_pickup_geo_unit", "booking", type_="foreignkey")
    op.drop_index(op.f("ix_booking_pricing_profile_version_id"), table_name="booking")
    op.drop_index(op.f("ix_booking_pricing_profile_id"), table_name="booking")
    op.drop_index(op.f("ix_booking_dropoff_geo_unit_id"), table_name="booking")
    op.drop_index(op.f("ix_booking_pickup_geo_unit_id"), table_name="booking")
    op.drop_column("booking", "price_breakdown_json")
    op.drop_column("booking", "price_amount")
    op.drop_column("booking", "pricing_profile_version_id")
    op.drop_column("booking", "pricing_profile_id")
    op.drop_column("booking", "dropoff_zip")
    op.drop_column("booking", "pickup_zip")
    op.drop_column("booking", "dropoff_geo_unit_id")
    op.drop_column("booking", "pickup_geo_unit_id")

    op.drop_index("ix_dispatch_offer_booking_status", table_name="dispatch_offer")
    op.drop_index(op.f("ix_dispatch_offer_company_id"), table_name="dispatch_offer")
    op.drop_index(op.f("ix_dispatch_offer_booking_id"), table_name="dispatch_offer")
    op.drop_table("dispatch_offer")

    op.drop_constraint("fk_pricing_profile_current_version", "pricing_profile", type_="foreignkey")
    op.drop_index("ix_pricing_profile_version_profile", table_name="pricing_profile_version")
    op.drop_index(op.f("ix_pricing_profile_version_pricing_profile_id"), table_name="pricing_profile_version")
    op.drop_table("pricing_profile_version")
    op.drop_index("ix_pricing_profile_company_active", table_name="pricing_profile")
    op.drop_index(op.f("ix_pricing_profile_company_id"), table_name="pricing_profile")
    op.drop_table("pricing_profile")

    op.drop_index("ix_service_area_company_active", table_name="service_area")
    op.drop_index(op.f("ix_service_area_geo_unit_id"), table_name="service_area")
    op.drop_index(op.f("ix_service_area_company_id"), table_name="service_area")
    op.drop_table("service_area")

    op.drop_index("ix_geo_unit_type_parent", table_name="geo_unit")
    op.drop_index(op.f("ix_geo_unit_parent_id"), table_name="geo_unit")
    op.drop_index(op.f("ix_geo_unit_code"), table_name="geo_unit")
    op.drop_table("geo_unit")

    bind = op.get_bind()
    dispatch_offer_status.drop(bind, checkfirst=True)
    pricing_model_type.drop(bind, checkfirst=True)
    service_coverage_mode.drop(bind, checkfirst=True)
    geo_unit_type.drop(bind, checkfirst=True)
