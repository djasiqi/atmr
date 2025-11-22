"""add timezone to driver_shift

Revision ID: 6d50ede67619
Revises: a452a931f0b7
Create Date: 2025-10-09 12:25:47.658343

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "6d50ede67619"
down_revision = "a452a931f0b7"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    table_names = inspector.get_table_names()

    if "company_planning_settings" not in table_names:
        op.create_table(
            "company_planning_settings",
            sa.Column("company_id", sa.Integer(), nullable=False),
            sa.Column(
                "settings", sa.JSON(), server_default=sa.text("'{}'"), nullable=False
            ),
            sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("company_id"),
        )

    if "driver_preference" not in table_names:
        op.create_table(
            "driver_preference",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("company_id", sa.Integer(), nullable=False),
            sa.Column("driver_id", sa.Integer(), nullable=False),
            sa.Column(
                "mornings_pref", sa.Boolean(), server_default="false", nullable=False
            ),
            sa.Column(
                "evenings_pref", sa.Boolean(), server_default="false", nullable=False
            ),
            sa.Column(
                "forbidden_windows",
                sa.JSON(),
                server_default=sa.text("'[]'"),
                nullable=False,
            ),
            sa.Column(
                "weekend_rotation_weight",
                sa.Integer(),
                server_default="0",
                nullable=False,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

        with op.batch_alter_table("driver_preference", schema=None) as batch_op:
            batch_op.create_index(
                batch_op.f("ix_driver_preference_company_id"),
                ["company_id"],
                unique=False,
            )
            batch_op.create_index(
                batch_op.f("ix_driver_preference_driver_id"),
                ["driver_id"],
                unique=False,
            )
    else:
        existing_indexes = {
            idx["name"] for idx in inspector.get_indexes("driver_preference")
        }
        indexes_to_create = [
            ("ix_driver_preference_company_id", ["company_id"]),
            ("ix_driver_preference_driver_id", ["driver_id"]),
        ]
        to_create = [
            item for item in indexes_to_create if item[0] not in existing_indexes
        ]
        if to_create:
            with op.batch_alter_table("driver_preference", schema=None) as batch_op:
                for name, columns in to_create:
                    batch_op.create_index(name, columns, unique=False)

    driver_shift_columns = {
        col["name"] for col in inspector.get_columns("driver_shift")
    }
    columns_to_add = [
        (
            "timezone",
            sa.String(length=64),
            {"server_default": "Europe/Zurich", "nullable": False},
        ),
        ("site", sa.String(length=120), {"nullable": True}),
        ("zone", sa.String(length=120), {"nullable": True}),
        ("client_ref", sa.String(length=120), {"nullable": True}),
        ("pay_code", sa.String(length=50), {"nullable": True}),
        ("vehicle_id", sa.Integer(), {"nullable": True}),
        ("notes_internal", sa.Text(), {"nullable": True}),
        ("notes_employee", sa.Text(), {"nullable": True}),
        ("updated_by_user_id", sa.Integer(), {"nullable": True}),
        ("version", sa.Integer(), {"server_default": "1", "nullable": False}),
        (
            "compliance_flags",
            sa.JSON(),
            {"server_default": sa.text("'[]'"), "nullable": False},
        ),
    ]

    indexes_existing = {idx["name"] for idx in inspector.get_indexes("driver_shift")}
    fk_info = inspector.get_foreign_keys("driver_shift")

    needs_index_vehicle = "ix_driver_shift_vehicle_id" not in indexes_existing
    has_fk_vehicle = any(
        fk["referred_table"] == "vehicle"
        and fk["constrained_columns"] == ["vehicle_id"]
        for fk in fk_info
    )
    has_fk_updated_by = any(
        fk["referred_table"] == "user"
        and fk["constrained_columns"] == ["updated_by_user_id"]
        for fk in fk_info
    )

    columns_missing = [
        (name, type_, kwargs)
        for name, type_, kwargs in columns_to_add
        if name not in driver_shift_columns
    ]

    if (
        columns_missing
        or needs_index_vehicle
        or not has_fk_vehicle
        or not has_fk_updated_by
    ):
        with op.batch_alter_table("driver_shift", schema=None) as batch_op:
            for name, type_, kwargs in columns_missing:
                batch_op.add_column(sa.Column(name, type_, **kwargs))

            if needs_index_vehicle:
                batch_op.create_index(
                    batch_op.f("ix_driver_shift_vehicle_id"),
                    ["vehicle_id"],
                    unique=False,
                )

            if not has_fk_vehicle:
                batch_op.create_foreign_key(
                    None, "vehicle", ["vehicle_id"], ["id"], ondelete="SET NULL"
                )

            if not has_fk_updated_by:
                batch_op.create_foreign_key(
                    None, "user", ["updated_by_user_id"], ["id"], ondelete="SET NULL"
                )

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("driver_shift", schema=None) as batch_op:
        batch_op.drop_constraint(None, type_="foreignkey")
        batch_op.drop_constraint(None, type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_driver_shift_vehicle_id"))
        batch_op.drop_column("compliance_flags")
        batch_op.drop_column("version")
        batch_op.drop_column("updated_by_user_id")
        batch_op.drop_column("notes_employee")
        batch_op.drop_column("notes_internal")
        batch_op.drop_column("vehicle_id")
        batch_op.drop_column("pay_code")
        batch_op.drop_column("client_ref")
        batch_op.drop_column("zone")
        batch_op.drop_column("site")
        batch_op.drop_column("timezone")

    with op.batch_alter_table("driver_preference", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_driver_preference_driver_id"))
        batch_op.drop_index(batch_op.f("ix_driver_preference_company_id"))

    op.drop_table("driver_preference")
    op.drop_table("company_planning_settings")
    # ### end Alembic commands ###
