"""add_ab_test_result_table

Création de la table ab_test_result pour stocker les résultats
des tests A/B comparant ML vs Heuristique.

Revision ID: 97c8d4f1e5a3
Revises: 156c2b818038
Create Date: 2025-10-20 18:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "97c8d4f1e5a3"
down_revision = "156c2b818038"
branch_labels = None
depends_on = None


def upgrade():
    # Créer table ab_test_result
    op.create_table(
        "ab_test_result",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("test_timestamp", sa.DateTime(), nullable=False),
        # Prédiction ML
        sa.Column("ml_delay_minutes", sa.Float(), nullable=False),
        sa.Column("ml_confidence", sa.Float(), nullable=True),
        sa.Column("ml_risk_level", sa.String(length=20), nullable=True),
        sa.Column("ml_prediction_time_ms", sa.Float(), nullable=False),
        sa.Column("ml_weather_factor", sa.Float(), nullable=True),
        # Prédiction Heuristique
        sa.Column("heuristic_delay_minutes", sa.Float(), nullable=False),
        sa.Column("heuristic_prediction_time_ms", sa.Float(), nullable=False),
        # Comparaison
        sa.Column("difference_minutes", sa.Float(), nullable=False),
        sa.Column("ml_faster", sa.Boolean(), nullable=False),
        sa.Column("speed_advantage_ms", sa.Float(), nullable=False),
        # Résultat réel (optionnel)
        sa.Column("actual_delay_minutes", sa.Float(), nullable=True),
        sa.Column("ml_error", sa.Float(), nullable=True),
        sa.Column("heuristic_error", sa.Float(), nullable=True),
        sa.Column("ml_winner", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"]),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"]),
    )

    # Index pour performance
    op.create_index(
        "ix_ab_test_result_booking_id", "ab_test_result", ["booking_id"], unique=False
    )
    op.create_index(
        "ix_ab_test_result_driver_id", "ab_test_result", ["driver_id"], unique=False
    )
    op.create_index(
        "ix_ab_test_result_test_timestamp",
        "ab_test_result",
        ["test_timestamp"],
        unique=False,
    )

    # Index composite pour analyses temporelles
    op.create_index(
        "ix_ab_test_result_timestamp_winner",
        "ab_test_result",
        ["test_timestamp", "ml_winner"],
        unique=False,
    )


def downgrade():
    # Suppression des index
    op.drop_index("ix_ab_test_result_timestamp_winner", table_name="ab_test_result")
    op.drop_index("ix_ab_test_result_test_timestamp", table_name="ab_test_result")
    op.drop_index("ix_ab_test_result_driver_id", table_name="ab_test_result")
    op.drop_index("ix_ab_test_result_booking_id", table_name="ab_test_result")

    # Suppression de la table
    op.drop_table("ab_test_result")
