"""sync_models_with_db

Synchronise la base de données avec les modèles SQLAlchemy actuels.
Corrige les écarts détectés par Alembic autogenerate.

Revision ID: 68116559b15d
Revises: f7e8d9c0b1a2
Create Date: 2025-11-21 12:00:00.000000

"""

from contextlib import suppress

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "68116559b15d"
down_revision = "f7e8d9c0b1a2"
branch_labels = None
depends_on = None


def upgrade():
    """Synchronise la DB avec les modèles SQLAlchemy."""

    # 1. client.company_id : rendre nullable=True
    with op.batch_alter_table("client", schema=None) as batch_op:
        batch_op.alter_column("company_id", existing_type=sa.INTEGER(), nullable=True)

        # 2. Créer l'index unique conditionnel uq_client_user_no_company
        batch_op.create_index(
            "uq_client_user_no_company", ["user_id"], unique=True, postgresql_where=sa.text("company_id IS NULL")
        )

        # 3. Modifier la FK pour ondelete='SET NULL' au lieu de 'CASCADE'
        batch_op.drop_constraint("client_company_id_fkey", type_="foreignkey")
        batch_op.create_foreign_key(None, "company", ["company_id"], ["id"], ondelete="SET NULL")

    # 4. driver.driver_photo : changer de TEXT à String(500)
    with op.batch_alter_table("driver", schema=None) as batch_op:
        batch_op.alter_column(
            "driver_photo", existing_type=sa.TEXT(), type_=sa.String(length=500), existing_nullable=True
        )

    # 5. task_failure : supprimer les indexes first_seen et last_seen
    # (ils existent en DB mais ne sont pas dans le modèle)
    with op.batch_alter_table("task_failure", schema=None) as batch_op:
        # Vérifier si les indexes existent avant de les supprimer
        # (utiliser batch_op.f() pour générer le nom automatique si nécessaire)
        with suppress(Exception):
            batch_op.drop_index("ix_task_failure_first_seen")
        with suppress(Exception):
            batch_op.drop_index("ix_task_failure_last_seen")

    # 6. user.email : changer de VARCHAR(100) à String(255)
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.alter_column(
            "email", existing_type=sa.VARCHAR(length=100), type_=sa.String(length=255), existing_nullable=True
        )

        # 7. user.phone : changer de VARCHAR(20) à String(255)
        batch_op.alter_column(
            "phone", existing_type=sa.VARCHAR(length=20), type_=sa.String(length=255), existing_nullable=True
        )


def downgrade():
    """Restaure l'état précédent de la DB."""

    # 1. user.phone : restaurer VARCHAR(20)
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.alter_column(
            "phone", existing_type=sa.String(length=255), type_=sa.VARCHAR(length=20), existing_nullable=True
        )

        # 2. user.email : restaurer VARCHAR(100)
        batch_op.alter_column(
            "email", existing_type=sa.String(length=255), type_=sa.VARCHAR(length=100), existing_nullable=True
        )

    # 3. task_failure : restaurer les indexes
    with op.batch_alter_table("task_failure", schema=None) as batch_op:
        batch_op.create_index("ix_task_failure_last_seen", ["last_seen"], unique=False)
        batch_op.create_index("ix_task_failure_first_seen", ["first_seen"], unique=False)

    # 4. driver.driver_photo : restaurer TEXT
    with op.batch_alter_table("driver", schema=None) as batch_op:
        batch_op.alter_column(
            "driver_photo", existing_type=sa.String(length=500), type_=sa.TEXT(), existing_nullable=True
        )

    # 5. client : restaurer FK CASCADE et nullable=False
    with op.batch_alter_table("client", schema=None) as batch_op:
        batch_op.drop_constraint(None, type_="foreignkey")
        batch_op.create_foreign_key("client_company_id_fkey", "company", ["company_id"], ["id"], ondelete="CASCADE")
        batch_op.drop_index("uq_client_user_no_company", postgresql_where=sa.text("company_id IS NULL"))
        batch_op.alter_column("company_id", existing_type=sa.INTEGER(), nullable=False)
