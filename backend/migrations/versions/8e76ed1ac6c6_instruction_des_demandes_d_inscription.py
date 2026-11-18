"""instruction des demandes d inscription

Revision ID: 8e76ed1ac6c6
Revises: ea71f45bd483
Create Date: 2026-10-12 18:31:03.814413

"""

from alembic import op
import sqlalchemy as sa


revision = "8e76ed1ac6c6"
down_revision = "ea71f45bd483"
branch_labels = None
depends_on = None


def upgrade():
    # Conservé depuis l'autogénération : uniquement l'historique des dossiers.
    # Le reste du diff décrivait un écart historique de schéma, sans lien avec
    # cette instruction, et n'est pas appliqué.
    op.add_column(
        "organization_application_event",
        sa.Column(
            "visibility",
            sa.String(length=16),
            server_default="public",
            nullable=False,
        ),
    )
    op.add_column(
        "organization_application_event",
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "organization_application_event",
        sa.Column("note", sa.Text(), nullable=True),
    )
    op.create_foreign_key(
        "fk_org_application_event_actor_user_id",
        "organization_application_event",
        "user",
        ["actor_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "ck_org_application_event_visibility",
        "organization_application_event",
        "visibility IN ('public', 'internal')",
    )


def downgrade():
    op.drop_constraint(
        "ck_org_application_event_visibility",
        "organization_application_event",
        type_="check",
    )
    op.drop_constraint(
        "fk_org_application_event_actor_user_id",
        "organization_application_event",
        type_="foreignkey",
    )
    op.drop_column("organization_application_event", "note")
    op.drop_column("organization_application_event", "actor_user_id")
    op.drop_column("organization_application_event", "visibility")
