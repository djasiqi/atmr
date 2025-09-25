"""add billed_to fields to Booking and is_partner to Company

Revision ID: 41e1a4a51b50
Revises: 5c88abe5821e
Create Date: 2025-08-30 17:30:22.122319
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '41e1a4a51b50'
down_revision = '5c88abe5821e'
branch_labels = None
depends_on = None


def upgrade():
    # --- booking ---
    with op.batch_alter_table('booking', schema=None) as batch_op:
        # ajoute avec server_default pour éviter les NULL au déploiement
        batch_op.add_column(sa.Column('billed_to_type', sa.String(length=50), server_default='patient', nullable=False))
        batch_op.add_column(sa.Column('billed_to_company_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('billed_to_contact', sa.String(length=120), nullable=True))

        # IMPORTANT: nom explicite requis en batch-mode SQLite
        batch_op.create_foreign_key(
            'fk_booking_billed_company',
            'company',
            ['billed_to_company_id'],
            ['id'],
            ondelete='SET NULL'
        )

    # --- company ---
    with op.batch_alter_table('company', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_partner', sa.Boolean(), server_default=sa.text('0'), nullable=False))

    # (Optionnel) retirer les defaults pour les prochains inserts
    with op.batch_alter_table('booking', schema=None) as batch_op:
        batch_op.alter_column('billed_to_type', server_default=None)

    with op.batch_alter_table('company', schema=None) as batch_op:
        batch_op.alter_column('is_partner', server_default=None)


def downgrade():
    # --- booking ---
    with op.batch_alter_table('booking', schema=None) as batch_op:
        batch_op.drop_constraint('fk_booking_billed_company', type_='foreignkey')
        batch_op.drop_column('billed_to_contact')
        batch_op.drop_column('billed_to_company_id')
        batch_op.drop_column('billed_to_type')

    # --- company ---
    with op.batch_alter_table('company', schema=None) as batch_op:
        batch_op.drop_column('is_partner')
