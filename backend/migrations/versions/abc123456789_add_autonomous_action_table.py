"""add_autonomous_action_table

Création de la table autonomous_action pour tracer toutes les actions
automatiques du système (audit trail).

Permet de:
- Audit trail complet de toutes les actions autonomes
- Rate limiting (limiter actions par heure/jour)
- Monitoring et détection d'anomalies
- Compliance et traçabilité

Revision ID: abc123456789
Revises: 97c8d4f1e5a3
Create Date: 2025-10-20 22:30:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "abc123456789"
down_revision = "97c8d4f1e5a3"  # Après ab_test_result
branch_labels = None
depends_on = None


def upgrade():
    # Créer table autonomous_action
    op.create_table(
        "autonomous_action",
        sa.Column("id", sa.Integer(), nullable=False),
        # Identifiants
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=True),
        sa.Column("driver_id", sa.Integer(), nullable=True),
        # Type d'action
        sa.Column("action_type", sa.String(length=50), nullable=False),
        sa.Column("action_description", sa.String(length=500), nullable=False),
        sa.Column("action_data", sa.Text(), nullable=True),
        # Résultat
        sa.Column("success", sa.Boolean(), nullable=False, default=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        # Métriques
        sa.Column("execution_time_ms", sa.Float(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("expected_improvement_minutes", sa.Float(), nullable=True),
        # Contexte
        sa.Column("trigger_source", sa.String(length=100), nullable=True),
        # Sécurité / Review
        sa.Column("reviewed_by_admin", sa.Boolean(), nullable=False, default=False),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("admin_notes", sa.Text(), nullable=True),
        # Métadonnées
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        # Clés
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"]),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"]),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"]),
    )

    # Index pour performance (rate limiting, queries fréquentes)
    op.create_index("ix_autonomous_action_company_id", "autonomous_action", ["company_id"], unique=False)
    op.create_index("ix_autonomous_action_booking_id", "autonomous_action", ["booking_id"], unique=False)
    op.create_index("ix_autonomous_action_driver_id", "autonomous_action", ["driver_id"], unique=False)
    op.create_index("ix_autonomous_action_action_type", "autonomous_action", ["action_type"], unique=False)
    op.create_index("ix_autonomous_action_created_at", "autonomous_action", ["created_at"], unique=False)

    # Index composite pour rate limiting (comptage par heure/jour)
    op.create_index(
        "ix_autonomous_action_company_created",
        "autonomous_action",
        ["company_id", "created_at", "success"],
        unique=False,
    )

    # Index composite pour audit par type et date
    op.create_index(
        "ix_autonomous_action_type_created", "autonomous_action", ["action_type", "created_at"], unique=False
    )


def downgrade():
    # Supprimer index
    op.drop_index("ix_autonomous_action_type_created", table_name="autonomous_action")
    op.drop_index("ix_autonomous_action_company_created", table_name="autonomous_action")
    op.drop_index("ix_autonomous_action_created_at", table_name="autonomous_action")
    op.drop_index("ix_autonomous_action_action_type", table_name="autonomous_action")
    op.drop_index("ix_autonomous_action_driver_id", table_name="autonomous_action")
    op.drop_index("ix_autonomous_action_booking_id", table_name="autonomous_action")
    op.drop_index("ix_autonomous_action_company_id", table_name="autonomous_action")

    # Supprimer table
    op.drop_table("autonomous_action")
