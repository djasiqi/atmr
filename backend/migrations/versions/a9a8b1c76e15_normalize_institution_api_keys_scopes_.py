"""normalize institution_api_keys schema drift (scopes + key_prefix)

Aligne ``institution_api_keys`` sur la définition modèle/migration d'origine
lorsque certaines bases ont dérivé :
- ``scopes`` : ``text[]`` -> ``Text`` (JSON, cf. set_scopes/get_scopes).
- ``key_prefix`` : ``varchar(8)`` -> ``varchar(20)`` (cf. migration 20260204).
- colonnes parasites ``public_id`` / ``is_active`` : supprimées (absentes du
  modèle et de toute migration ; ``is_active`` est une @property Python).
Conversions idempotentes, sans effet sur les environnements déjà conformes.

Revision ID: a9a8b1c76e15
Revises: eb21b7cf8467
Create Date: 2026-06-16 17:04:15.851566

"""

from alembic import op
import sqlalchemy as sa


revision = "a9a8b1c76e15"
down_revision = "eb21b7cf8467"
branch_labels = None
depends_on = None


def upgrade():
    # Convertit scopes (text[]) -> text JSON uniquement si la colonne a dérivé.
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'institution_api_keys'
                      AND column_name = 'scopes'
                      AND data_type = 'ARRAY'
                ) THEN
                    ALTER TABLE institution_api_keys
                        ALTER COLUMN scopes DROP DEFAULT;
                    ALTER TABLE institution_api_keys
                        ALTER COLUMN scopes TYPE text
                        USING to_json(scopes)::text;
                    ALTER TABLE institution_api_keys
                        ALTER COLUMN scopes SET DEFAULT '[]';
                END IF;
            END $$;
            """
        )
    )
    # Élargit key_prefix si une base a dérivé vers une longueur < 20 (modèle = 20).
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'institution_api_keys'
                      AND column_name = 'key_prefix'
                      AND character_maximum_length IS NOT NULL
                      AND character_maximum_length < 20
                ) THEN
                    ALTER TABLE institution_api_keys
                        ALTER COLUMN key_prefix TYPE varchar(20);
                END IF;
            END $$;
            """
        )
    )
    # Supprime les colonnes parasites (drift) absentes du modèle/migrations.
    op.execute(
        sa.text(
            "ALTER TABLE institution_api_keys DROP COLUMN IF EXISTS public_id;"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE institution_api_keys DROP COLUMN IF EXISTS is_active;"
        )
    )


def downgrade():
    # No-op volontaire : le type ``text[]`` était une dérive locale jamais définie
    # par une migration (la table a toujours été créée en ``Text`` JSON, cf.
    # 20260204_add_institution_api_keys). Réintroduire ``text[]`` au downgrade
    # recréerait l'incohérence avec le modèle ; on conserve donc ``Text``.
    pass
