"""Grille d'abonnement plateforme LIRIE — paliers par défaut (semi_auto, fully_auto, manual).

Les prix sont des exemples commerciaux type pilote ; les ajuster en base si besoin.

Revision ID: 20260401_seed_plat_sub_pricing
Revises: 20260331_plat_bill_v1
Create Date: 2026-04-01

"""

from __future__ import annotations

from alembic import op

revision = "20260401_seed_plat_sub_pricing"
down_revision = "20260331_plat_bill_v1"
branch_labels = None
depends_on = None

_SEED_PREFIX = "LIRIE V1 défaut"


def upgrade() -> None:
    op.execute(
        f"""
        INSERT INTO platform_subscription_pricing
            (dispatch_mode, volume_min, volume_max, price_monthly, label)
        SELECT * FROM (VALUES
            ('semi_auto'::varchar(16), 0::integer, 200::integer, 79.00::numeric,
             '{_SEED_PREFIX} semi_auto 0-200'),
            ('semi_auto', 201, 500, 149.00::numeric,
             '{_SEED_PREFIX} semi_auto 201-500'),
            ('semi_auto', 501, NULL::integer, 249.00::numeric,
             '{_SEED_PREFIX} semi_auto 501+'),
            ('fully_auto', 0, 200, 149.00::numeric,
             '{_SEED_PREFIX} fully_auto 0-200'),
            ('fully_auto', 201, 500, 299.00::numeric,
             '{_SEED_PREFIX} fully_auto 201-500'),
            ('fully_auto', 501, NULL::integer, 499.00::numeric,
             '{_SEED_PREFIX} fully_auto 501+'),
            ('manual', 0, NULL::integer, 0.00::numeric,
             '{_SEED_PREFIX} manual (0 CHF)')
        ) AS v(dispatch_mode, volume_min, volume_max, price_monthly, label)
        WHERE NOT EXISTS (
            SELECT 1 FROM platform_subscription_pricing p
            WHERE p.label LIKE '{_SEED_PREFIX}%'
            LIMIT 1
        );
        """
    )


def downgrade() -> None:
    op.execute(
        f"""
        DELETE FROM platform_subscription_pricing
        WHERE label LIKE '{_SEED_PREFIX}%';
        """
    )
