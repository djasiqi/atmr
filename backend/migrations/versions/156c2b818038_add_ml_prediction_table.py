"""add_ml_prediction_table

Création de la table ml_prediction pour stocker les prédictions ML
et permettre le monitoring en temps réel.

Permet de:
- Tracker toutes les prédictions ML
- Comparer prédictions vs réalité
- Calculer métriques (MAE, R²)
- Détecter anomalies et drift

Revision ID: 156c2b818038
Revises: b559b3ef7a75
Create Date: 2025-10-20 17:46:34.413500

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "156c2b818038"
down_revision = "b559b3ef7a75"
branch_labels = None
depends_on = None


def upgrade():
    # Créer table ml_prediction
    op.create_table(
        "ml_prediction",
        sa.Column("id", sa.Integer(), nullable=False),

        # Identifiants
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=True),
        sa.Column("request_id", sa.String(length=0.100), nullable=True),

        # Prédiction ML
        sa.Column("predicted_delay_minutes", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column("contributing_factors", sa.Text(), nullable=True),

        # Contexte prédiction
        sa.Column("model_version", sa.String(length=50), nullable=True),
        sa.Column("prediction_time_ms", sa.Float(), nullable=True),
        sa.Column("feature_flag_enabled", sa.Boolean(), nullable=True),
        sa.Column("traffic_percentage", sa.Integer(), nullable=True),

        # Résultat réel
        sa.Column("actual_delay_minutes", sa.Float(), nullable=True),
        sa.Column("actual_pickup_at", sa.DateTime(), nullable=True),
        sa.Column("actual_dropoff_at", sa.DateTime(), nullable=True),

        # Métriques calculées
        sa.Column("prediction_error", sa.Float(), nullable=True),
        sa.Column("is_accurate", sa.Boolean(), nullable=True),

        # Métadonnées
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),

        # Clés
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"] ),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"] ),
    )

    # Index pour performance
    op.create_index("ix_ml_prediction_booking_id", "ml_prediction", ["booking_id"], unique=False)
    op.create_index("ix_ml_prediction_driver_id", "ml_prediction", ["driver_id"], unique=False)
    op.create_index("ix_ml_prediction_request_id", "ml_prediction", ["request_id"], unique=False)
    op.create_index("ix_ml_prediction_created_at", "ml_prediction", ["created_at"], unique=False)

    # Index composite pour queries fréquentes
    op.create_index(
        "ix_ml_prediction_created_actual",
        "ml_prediction",
        ["created_at", "actual_delay_minutes"],
        unique=False
    )


def downgrade():
    # Supprimer index
    op.drop_index("ix_ml_prediction_created_actual", table_name="ml_prediction")
    op.drop_index("ix_ml_prediction_created_at", table_name="ml_prediction")
    op.drop_index("ix_ml_prediction_request_id", table_name="ml_prediction")
    op.drop_index("ix_ml_prediction_driver_id", table_name="ml_prediction")
    op.drop_index("ix_ml_prediction_booking_id", table_name="ml_prediction")

    # Supprimer table
    op.drop_table("ml_prediction")
