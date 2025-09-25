"""add company legal/billing fields, vehicle table, and enrich client

Revision ID: b6e2a5f19332
Revises: 41e1a4a51b50
Create Date: 2025-08-30 18:08:12.736850
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b6e2a5f19332'
down_revision = '41e1a4a51b50'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # --- VEHICLE table (idempotent) ---
    if not insp.has_table('vehicle'):
        op.create_table(
            'vehicle',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('company_id', sa.Integer(), sa.ForeignKey('company.id', name='fk_vehicle_company', ondelete='CASCADE'), nullable=False),
            sa.Column('model', sa.String(length=120), nullable=False),
            sa.Column('license_plate', sa.String(length=20), nullable=False),
            sa.Column('year', sa.Integer(), nullable=True),
            sa.Column('vin', sa.String(length=32), nullable=True),
            sa.Column('seats', sa.Integer(), nullable=True),
            sa.Column('wheelchair_accessible', sa.Boolean(), nullable=False, server_default=sa.text('0')),
            sa.Column('insurance_expires_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('inspection_expires_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('1')),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint('company_id', 'license_plate', name='uq_company_plate'),
        )

    # index sur vehicle.company_id (si absent)
    vehicle_indexes = {ix['name'] for ix in insp.get_indexes('vehicle')} if insp.has_table('vehicle') else set()
    if 'ix_vehicle_company_id' not in vehicle_indexes:
        op.create_index('ix_vehicle_company_id', 'vehicle', ['company_id'], unique=False)

    # --- CLIENT: domicile/access/gp/default_billing ---
    with op.batch_alter_table('client', schema=None) as batch_op:
        batch_op.add_column(sa.Column('domicile_address', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('domicile_zip', sa.String(length=10), nullable=True))
        batch_op.add_column(sa.Column('domicile_city', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('door_code', sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column('floor', sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column('access_notes', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('gp_name', sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column('gp_phone', sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column('default_billed_to_type', sa.String(length=50), server_default='patient', nullable=False))
        batch_op.add_column(sa.Column('default_billed_to_company_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('default_billed_to_contact', sa.String(length=120), nullable=True))
        # 🔑 nommer la FK (sinon SQLite râle)
        batch_op.create_foreign_key(
            'fk_client_default_billed_company',
            'company',
            ['default_billed_to_company_id'],
            ['id'],
            ondelete='SET NULL'
        )

    # Retirer le default pour les prochains inserts
    with op.batch_alter_table('client', schema=None) as batch_op:
        batch_op.alter_column('default_billed_to_type', server_default=None)

    # --- COMPANY: legal/billing + domicile ---
    with op.batch_alter_table('company', schema=None) as batch_op:
        batch_op.add_column(sa.Column('domicile_address_line1', sa.String(length=200), nullable=True))
        batch_op.add_column(sa.Column('domicile_address_line2', sa.String(length=200), nullable=True))
        batch_op.add_column(sa.Column('domicile_zip', sa.String(length=10), nullable=True))
        batch_op.add_column(sa.Column('domicile_city', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('domicile_country', sa.String(length=2), nullable=True, server_default='CH'))
        batch_op.add_column(sa.Column('iban', sa.String(length=34), nullable=True))
        batch_op.add_column(sa.Column('uid_ide', sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column('billing_email', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('billing_notes', sa.Text(), nullable=True))

    # indexes sur company (si absents)
    company_indexes = {ix['name'] for ix in insp.get_indexes('company')}
    if 'ix_company_iban' not in company_indexes:
        op.create_index('ix_company_iban', 'company', ['iban'], unique=False)
    if 'ix_company_uid_ide' not in company_indexes:
        op.create_index('ix_company_uid_ide', 'company', ['uid_ide'], unique=False)

    # retirer le default 'CH' si tu ne veux pas le conserver
    with op.batch_alter_table('company', schema=None) as batch_op:
        batch_op.alter_column('domicile_country', server_default=None)


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # --- COMPANY ---
    if 'ix_company_uid_ide' in {ix['name'] for ix in insp.get_indexes('company')}:
        op.drop_index('ix_company_uid_ide', table_name='company')
    if 'ix_company_iban' in {ix['name'] for ix in insp.get_indexes('company')}:
        op.drop_index('ix_company_iban', table_name='company')

    with op.batch_alter_table('company', schema=None) as batch_op:
        for col in [
            'billing_notes', 'billing_email', 'uid_ide', 'iban',
            'domicile_country', 'domicile_city', 'domicile_zip',
            'domicile_address_line2', 'domicile_address_line1'
        ]:
            if col in {c['name'] for c in insp.get_columns('company')}:
                batch_op.drop_column(col)

    # --- CLIENT ---
    with op.batch_alter_table('client', schema=None) as batch_op:
        # drop FK si existe
        try:
            batch_op.drop_constraint('fk_client_default_billed_company', type_='foreignkey')
        except Exception:
            pass
        for col in [
            'default_billed_to_contact', 'default_billed_to_company_id', 'default_billed_to_type',
            'gp_phone', 'gp_name', 'access_notes', 'floor', 'door_code',
            'domicile_city', 'domicile_zip', 'domicile_address'
        ]:
            if col in {c['name'] for c in insp.get_columns('client')}:
                batch_op.drop_column(col)

    # --- VEHICLE ---
    if insp.has_table('vehicle'):
        # drop index si présent
        if 'ix_vehicle_company_id' in {ix['name'] for ix in insp.get_indexes('vehicle')}:
            op.drop_index('ix_vehicle_company_id', table_name='vehicle')
        op.drop_table('vehicle')
