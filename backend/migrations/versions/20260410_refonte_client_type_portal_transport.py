"""Refonte ClientType : PORTAL / TRANSPORT + management_mode.

Phase A de la refonte du modele client. Ajoute les nouvelles valeurs
PORTAL et TRANSPORT a l'enum client_type, cree l'enum management_mode,
migre les donnees existantes et conserve une colonne rollback.

Revision ID: 20260410_client_type_v2
Revises: 20260410_addr500
Create Date: 2026-04-10
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "20260410_client_type_v2"
down_revision = "20260410_addr500"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    conn = op.get_bind()

    # --- 1. Ajouter PORTAL et TRANSPORT a l'enum existant ---
    # ALTER TYPE ... ADD VALUE ne peut pas etre dans une transaction,
    # Alembic execute hors transaction par defaut pour ces operations.
    conn.execute(sa.text("ALTER TYPE client_type ADD VALUE IF NOT EXISTS 'PORTAL'"))
    conn.execute(sa.text("ALTER TYPE client_type ADD VALUE IF NOT EXISTS 'TRANSPORT'"))
    conn.execute(sa.text("COMMIT"))

    # --- 2. Creer l'enum management_mode ---
    management_mode_enum = sa.Enum(
        "SELF_SERVICE", "MANAGED", "CORPORATE", name="management_mode"
    )
    management_mode_enum.create(conn, checkfirst=True)

    # --- 3. Ajouter les nouvelles colonnes ---
    op.add_column(
        "client",
        sa.Column("management_mode", sa.Enum(
            "SELF_SERVICE", "MANAGED", "CORPORATE", name="management_mode",
            create_type=False,
        ), nullable=True),
    )
    op.add_column(
        "client",
        sa.Column("_old_client_type", sa.String(20), nullable=True),
    )

    # --- 4. Sauvegarder l'ancien type ---
    conn.execute(sa.text("UPDATE client SET _old_client_type = client_type::text"))

    # --- 5. Detecter et journaliser les anomalies ---
    anomalies = conn.execute(sa.text(
        "SELECT id, client_type::text, company_id, user_id, contact_email "
        "FROM client "
        "WHERE client_type::text IN ('SELF_SERVICE', 'CORPORATE') "
        "AND company_id IS NULL"
    )).fetchall()

    if anomalies:
        logger.warning(
            "=== ANOMALIES DE DONNEES DETECTEES : %d client(s) avec type "
            "entreprise mais sans company_id ===",
            len(anomalies),
        )
        for row in anomalies:
            logger.warning(
                "  Anomalie: id=%s, ancien_type=%s, company_id=%s, "
                "user_id=%s, email=%s -> corrige en PORTAL",
                row[0], row[1], row[2], row[3], row[4],
            )
    else:
        logger.info("Aucune anomalie de donnees detectee.")

    # --- 6. Migration des donnees ---
    # Tous les clients sans company_id -> PORTAL
    result_portal = conn.execute(sa.text(
        "UPDATE client SET client_type = 'PORTAL', management_mode = NULL "
        "WHERE company_id IS NULL"
    ))
    logger.info("Migre %d client(s) vers PORTAL.", result_portal.rowcount)

    # Clients avec company_id : type selon l'ancien type
    result_managed = conn.execute(sa.text(
        "UPDATE client SET client_type = 'TRANSPORT', management_mode = 'MANAGED' "
        "WHERE company_id IS NOT NULL AND _old_client_type = 'PRIVATE'"
    ))
    logger.info("Migre %d client(s) vers TRANSPORT/MANAGED.", result_managed.rowcount)

    result_ss = conn.execute(sa.text(
        "UPDATE client SET client_type = 'TRANSPORT', management_mode = 'SELF_SERVICE' "
        "WHERE company_id IS NOT NULL AND _old_client_type = 'SELF_SERVICE'"
    ))
    logger.info("Migre %d client(s) vers TRANSPORT/SELF_SERVICE.", result_ss.rowcount)

    result_corp = conn.execute(sa.text(
        "UPDATE client SET client_type = 'TRANSPORT', management_mode = 'CORPORATE' "
        "WHERE company_id IS NOT NULL AND _old_client_type = 'CORPORATE'"
    ))
    logger.info("Migre %d client(s) vers TRANSPORT/CORPORATE.", result_corp.rowcount)

    # --- 7. Mettre a jour le server_default ---
    op.alter_column(
        "client", "client_type",
        server_default="PORTAL",
    )

    # --- 8. Rapport final ---
    final = conn.execute(sa.text(
        "SELECT client_type::text, COUNT(*) FROM client GROUP BY client_type ORDER BY 1"
    )).fetchall()
    logger.info("=== Rapport post-migration ===")
    for ct, count in final:
        logger.info("  %s : %d client(s)", ct, count)


def downgrade() -> None:
    conn = op.get_bind()

    # Restaurer les anciennes valeurs depuis la colonne rollback
    conn.execute(sa.text(
        "UPDATE client SET client_type = _old_client_type::client_type "
        "WHERE _old_client_type IS NOT NULL"
    ))

    # Restaurer le server_default
    op.alter_column(
        "client", "client_type",
        server_default="PRIVATE",
    )

    # Supprimer les colonnes ajoutees
    op.drop_column("client", "_old_client_type")
    op.drop_column("client", "management_mode")

    # Supprimer l'enum management_mode
    sa.Enum(name="management_mode").drop(conn, checkfirst=True)

    # Note : les valeurs PORTAL/TRANSPORT restent dans l'enum PostgreSQL
    # (ALTER TYPE ADD VALUE est irreversible). Sans impact fonctionnel.
