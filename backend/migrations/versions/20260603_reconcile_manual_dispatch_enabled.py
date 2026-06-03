"""Réconciliation invariant: MANUAL ⇒ dispatch_enabled=false.

Coupe `dispatch_enabled` pour toutes les sociétés actuellement en mode MANUAL
mais avec le dispatch (auto) activé. Cette combinaison incohérente ouvrait la
porte à des assignations automatiques en mode manuel.

Idempotent : ne touche que les lignes violant l'invariant.
"""

from __future__ import annotations

from alembic import op

revision = "20260603_manual_invariant"
down_revision = "20260601_device_health"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # L'enum dispatchmode est stocké en UPPERCASE en base (cf. migration
    # af5e460cd09e). On compare de façon robuste via UPPER() au cas où des
    # valeurs lowercase résiduelles subsisteraient.
    op.execute(
        "UPDATE company "
        "SET dispatch_enabled = false "
        "WHERE UPPER(dispatch_mode::text) = 'MANUAL' "
        "AND dispatch_enabled = true"
    )


def downgrade() -> None:
    # No-op volontaire : réactiver dispatch_enabled pour les sociétés MANUAL
    # réintroduirait précisément le bug que cette migration corrige, et l'état
    # antérieur (quelles sociétés étaient à true) n'est pas récupérable.
    pass
