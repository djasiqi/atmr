"""✅ S2: Encrypt IBAN fields in database (RGPD compliance)

Chiffre les champs IBAN dans les tables company et company_billing_settings
pour conformité RGPD.

Revision ID: 2100d6d88008
Revises: 311e1f6c9c9d
Create Date: 2025-12-09 00:08:31.345730

"""

import sqlalchemy as sa
from alembic import op

# Constantes pour éviter les valeurs magiques
IBAN_MAX_LENGTH = 34  # Longueur maximale d'un IBAN en clair
BILLING_IBAN_MAX_LENGTH = (
    50  # Longueur maximale d'un IBAN dans billing_settings (peut inclure QR-IBAN)
)

revision = "2100d6d88008"
down_revision = "311e1f6c9c9d"
branch_labels = None
depends_on = None


def upgrade():
    """Chiffre les IBAN existants et augmente la taille des colonnes."""
    # ✅ S2: Augmenter la taille des colonnes IBAN pour stocker le texte chiffré (base64)
    # Le texte chiffré peut être plus long que l'IBAN original (max 200 caractères)
    with op.batch_alter_table("company", schema=None) as batch_op:
        batch_op.alter_column(
            "iban",
            existing_type=sa.VARCHAR(length=34),
            type_=sa.String(length=200),
            existing_nullable=True,
        )
        # Supprimer l'index sur iban car il n'est plus utile (données chiffrées)
        # Vérifier si l'index existe avant de le supprimer
        conn = op.get_bind()
        result = conn.execute(
            sa.text(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE tablename = 'company'
                    AND indexname = 'ix_company_iban'
                )
                """
            )
        ).scalar()
        if result:
            batch_op.drop_index(batch_op.f("ix_company_iban"))

    with op.batch_alter_table("company_billing_settings", schema=None) as batch_op:
        batch_op.alter_column(
            "iban",
            existing_type=sa.VARCHAR(length=50),
            type_=sa.String(length=200),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "qr_iban",
            existing_type=sa.VARCHAR(length=50),
            type_=sa.String(length=200),
            existing_nullable=True,
        )

    # ✅ S2: Chiffrer les IBAN existants
    # Note: Cette opération nécessite que l'EncryptionService soit disponible
    # et que MASTER_ENCRYPTION_KEY soit configuré
    connection = op.get_bind()

    try:
        # Importer EncryptionService dans le contexte de la migration
        import sys
        from pathlib import Path

        # Ajouter le chemin backend au PYTHONPATH pour les imports
        backend_path = Path(__file__).parent.parent.parent
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))

        from security.crypto import get_encryption_service

        encryption_service = get_encryption_service()

        # Chiffrer les IBAN dans la table company
        companies = connection.execute(
            sa.text(
                "SELECT id, iban FROM company WHERE iban IS NOT NULL AND iban != ''"
            )
        ).fetchall()

        encrypted_count = 0
        for company_id, iban_plain in companies:
            try:
                # Vérifier si l'IBAN est déjà chiffré (commence par base64)
                # Les IBAN chiffrés sont en base64 et ne ressemblent pas à un IBAN
                if iban_plain and len(iban_plain) > IBAN_MAX_LENGTH:
                    # Probablement déjà chiffré, ignorer
                    continue

                # Chiffrer l'IBAN
                iban_encrypted = encryption_service.encrypt_field(
                    iban_plain.strip().upper()
                )
                # Mettre à jour la base de données
                connection.execute(
                    sa.text("UPDATE company SET iban = :encrypted WHERE id = :id"),
                    {"encrypted": iban_encrypted, "id": company_id},
                )
                encrypted_count += 1
            except Exception as e:
                # Logger l'erreur mais continuer avec les autres
                print(f"⚠️ Erreur chiffrement IBAN company_id={company_id}: {e}")

        print(f"✅ {encrypted_count} IBAN chiffrés dans la table company")

        # Chiffrer les IBAN dans la table company_billing_settings
        billing_settings = connection.execute(
            sa.text(
                "SELECT id, iban, qr_iban FROM company_billing_settings "
                "WHERE (iban IS NOT NULL AND iban != '') OR (qr_iban IS NOT NULL AND qr_iban != '')"
            )
        ).fetchall()

        encrypted_iban_count = 0
        encrypted_qr_iban_count = 0
        for setting_id, iban_plain, qr_iban_plain in billing_settings:
            try:
                # Chiffrer l'IBAN si présent et pas déjà chiffré
                if iban_plain and len(iban_plain) <= BILLING_IBAN_MAX_LENGTH:
                    iban_encrypted = encryption_service.encrypt_field(
                        iban_plain.strip().upper()
                    )
                    connection.execute(
                        sa.text(
                            "UPDATE company_billing_settings SET iban = :encrypted WHERE id = :id"
                        ),
                        {"encrypted": iban_encrypted, "id": setting_id},
                    )
                    encrypted_iban_count += 1

                # Chiffrer le QR-IBAN si présent et pas déjà chiffré
                if qr_iban_plain and len(qr_iban_plain) <= BILLING_IBAN_MAX_LENGTH:
                    qr_iban_encrypted = encryption_service.encrypt_field(
                        qr_iban_plain.strip().upper()
                    )
                    connection.execute(
                        sa.text(
                            "UPDATE company_billing_settings SET qr_iban = :encrypted WHERE id = :id"
                        ),
                        {"encrypted": qr_iban_encrypted, "id": setting_id},
                    )
                    encrypted_qr_iban_count += 1
            except Exception as e:
                # Logger l'erreur mais continuer avec les autres
                print(
                    f"⚠️ Erreur chiffrement IBAN billing_settings_id={setting_id}: {e}"
                )

        print(
            f"✅ {encrypted_iban_count} IBAN et {encrypted_qr_iban_count} QR-IBAN chiffrés "
            f"dans la table company_billing_settings"
        )

        # Ne jamais commit explicitement dans une migration Alembic:
        # la transaction est gérée par Alembic, sinon le suivi de version peut diverger.

    except Exception as e:
        print(f"⚠️ Erreur lors du chiffrement des IBAN: {e}")
        print(
            "⚠️ Les colonnes ont été agrandies mais les données n'ont pas été chiffrées."
        )
        print(
            "⚠️ Vous devrez exécuter manuellement le chiffrement des données existantes."
        )
        # Ne pas lever d'exception pour ne pas bloquer la migration
        # Les données resteront en clair jusqu'à ce qu'elles soient mises à jour


def downgrade():
    """⚠️ ATTENTION: Le downgrade ne peut pas déchiffrer les données.

    Cette opération est irréversible sans la clé de chiffrement.
    On restaure uniquement la taille des colonnes.
    """
    # ⚠️ S2: Ne pas déchiffrer (irréversible sans clé)
    # Restaurer uniquement la taille des colonnes
    with op.batch_alter_table("company_billing_settings", schema=None) as batch_op:
        batch_op.alter_column(
            "qr_iban",
            existing_type=sa.String(length=200),
            type_=sa.VARCHAR(length=50),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "iban",
            existing_type=sa.String(length=200),
            type_=sa.VARCHAR(length=50),
            existing_nullable=True,
        )

    with op.batch_alter_table("company", schema=None) as batch_op:
        # Recréer l'index (même si les données sont chiffrées)
        batch_op.create_index(batch_op.f("ix_company_iban"), ["iban"], unique=False)
        batch_op.alter_column(
            "iban",
            existing_type=sa.String(length=200),
            type_=sa.VARCHAR(length=34),
            existing_nullable=True,
        )
