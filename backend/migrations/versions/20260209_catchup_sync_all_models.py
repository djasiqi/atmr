"""catchup: sync all models with database (idempotent)

Revision ID: 20260209_catchup
Revises: 055c847af0bf
Create Date: 2026-02-09

Migration idempotente qui rattrape l'ecart entre les modeles SQLAlchemy
et le schema de la base de donnees. Utilise IF NOT EXISTS partout pour
etre safe en local (tables billing manquantes) ET en production
(tables institution manquantes).

NE SUPPRIME AUCUNE COLONNE/TABLE pour eviter toute perte de donnees.
"""

from alembic import op


revision = "20260209_catchup"
down_revision = "055c847af0bf"
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# Toutes les operations utilisent du SQL brut avec IF NOT EXISTS
# pour etre 100% idempotentes : la migration peut etre relancee sans risque.
# ---------------------------------------------------------------------------

# ===== TABLES =====

CREATE_TABLES = [
    # -- billing_parties (existe en prod, pas en local)
    """
    CREATE TABLE IF NOT EXISTS billing_parties (
        id SERIAL PRIMARY KEY,
        company_id INTEGER NOT NULL REFERENCES company(id) ON DELETE CASCADE,
        type VARCHAR(50) NOT NULL,
        display_name VARCHAR(255) NOT NULL,
        billing_address TEXT,
        contact_email VARCHAR(255),
        contact_phone VARCHAR(50),
        external_ref VARCHAR(120),
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # -- client_billing_parties
    """
    CREATE TABLE IF NOT EXISTS client_billing_parties (
        id SERIAL PRIMARY KEY,
        client_id INTEGER NOT NULL REFERENCES client(id) ON DELETE CASCADE,
        billing_party_id INTEGER NOT NULL REFERENCES billing_parties(id) ON DELETE CASCADE,
        role VARCHAR(50),
        contact_name VARCHAR(120),
        contact_email VARCHAR(255),
        contact_phone VARCHAR(50),
        client_reference VARCHAR(80),
        is_default BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # -- billing_audit_logs
    """
    CREATE TABLE IF NOT EXISTS billing_audit_logs (
        id SERIAL PRIMARY KEY,
        company_id INTEGER NOT NULL REFERENCES company(id) ON DELETE CASCADE,
        booking_id INTEGER NOT NULL REFERENCES booking(id) ON DELETE CASCADE,
        actor_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
        action TEXT NOT NULL,
        reason TEXT,
        before JSONB,
        after JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # -- company_billing_profile
    """
    CREATE TABLE IF NOT EXISTS company_billing_profile (
        id SERIAL PRIMARY KEY,
        company_id INTEGER NOT NULL UNIQUE REFERENCES company(id) ON DELETE CASCADE,
        legal_name VARCHAR(200) NOT NULL,
        brand_name VARCHAR(200),
        uid_ide VARCHAR(20) NOT NULL,
        street_name VARCHAR(70) NOT NULL,
        building_number VARCHAR(16) NOT NULL,
        postal_code VARCHAR(16) NOT NULL,
        city VARCHAR(35) NOT NULL,
        country_code VARCHAR(2) NOT NULL DEFAULT 'CH',
        billing_email VARCHAR(100) NOT NULL,
        billing_phone VARCHAR(20) NOT NULL,
        vat_registered BOOLEAN NOT NULL DEFAULT FALSE,
        vat_number VARCHAR(50),
        vat_rate NUMERIC(5,2),
        iban VARCHAR(200) NOT NULL,
        qr_iban VARCHAR(200),
        payment_reference_mode VARCHAR(10) NOT NULL DEFAULT 'SCOR',
        creditor_reference_base VARCHAR(20),
        payment_terms_days INTEGER NOT NULL DEFAULT 30,
        overdue_fee NUMERIC(10,2) NOT NULL DEFAULT 15.00,
        legal_footer TEXT,
        is_address_validated BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ
    )
    """,
    # -- client_stays
    """
    CREATE TABLE IF NOT EXISTS client_stays (
        id SERIAL PRIMARY KEY,
        client_id INTEGER NOT NULL REFERENCES client(id) ON DELETE CASCADE,
        company_id INTEGER NOT NULL REFERENCES company(id) ON DELETE CASCADE,
        start_date TIMESTAMPTZ NOT NULL,
        end_date TIMESTAMPTZ,
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        source VARCHAR(50),
        notes TEXT,
        created_by_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # -- clinic_billing_party_mappings
    """
    CREATE TABLE IF NOT EXISTS clinic_billing_party_mappings (
        id SERIAL PRIMARY KEY,
        company_id INTEGER NOT NULL REFERENCES company(id) ON DELETE CASCADE,
        clinic_company_id INTEGER NOT NULL REFERENCES company(id) ON DELETE CASCADE,
        billing_party_id INTEGER NOT NULL REFERENCES billing_parties(id) ON DELETE CASCADE,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_clinic_billing_party_mapping_company_clinic
            UNIQUE (company_id, clinic_company_id)
    )
    """,
    # -- device_tokens
    """
    CREATE TABLE IF NOT EXISTS device_tokens (
        id SERIAL PRIMARY KEY,
        driver_id INTEGER NOT NULL REFERENCES driver(id) ON DELETE CASCADE,
        token VARCHAR(255) NOT NULL,
        device_id VARCHAR(255),
        platform VARCHAR(20),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        is_active BOOLEAN NOT NULL DEFAULT TRUE
    )
    """,
    # -- partner_invoices
    """
    CREATE TABLE IF NOT EXISTS partner_invoices (
        id SERIAL PRIMARY KEY,
        partnership_id INTEGER NOT NULL REFERENCES partnerships(id) ON DELETE CASCADE,
        executing_company_id INTEGER NOT NULL REFERENCES company(id) ON DELETE CASCADE,
        period_year INTEGER NOT NULL,
        period_month INTEGER NOT NULL,
        invoice_number VARCHAR(100) NOT NULL UNIQUE,
        subtotal_amount NUMERIC(10,2) NOT NULL DEFAULT 0,
        vat_amount NUMERIC(10,2) NOT NULL DEFAULT 0,
        total_amount NUMERIC(10,2) NOT NULL DEFAULT 0,
        amount_paid NUMERIC(10,2) NOT NULL DEFAULT 0,
        credit_balance NUMERIC(10,2) NOT NULL DEFAULT 0,
        tip_amount NUMERIC(10,2) NOT NULL DEFAULT 0,
        currency VARCHAR(3) NOT NULL DEFAULT 'CHF',
        status VARCHAR(20) NOT NULL DEFAULT 'draft',
        issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        due_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        paid_at TIMESTAMPTZ,
        sent_at TIMESTAMPTZ,
        pdf_url VARCHAR(500),
        notes VARCHAR(1000)
    )
    """,
    # -- transport_vouchers
    """
    CREATE TABLE IF NOT EXISTS transport_vouchers (
        id SERIAL PRIMARY KEY,
        company_id INTEGER NOT NULL REFERENCES company(id) ON DELETE CASCADE,
        client_id INTEGER NOT NULL REFERENCES client(id) ON DELETE CASCADE,
        booking_id INTEGER REFERENCES booking(id) ON DELETE SET NULL,
        billing_party_id INTEGER REFERENCES billing_parties(id) ON DELETE SET NULL,
        type VARCHAR(50) NOT NULL DEFAULT 'clinic',
        status VARCHAR(50) NOT NULL DEFAULT 'draft',
        valid_from TIMESTAMPTZ,
        valid_to TIMESTAMPTZ,
        external_ref VARCHAR(255),
        notes TEXT,
        validated_by_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
        validated_at TIMESTAMPTZ,
        created_by_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # -- transport_voucher_files
    """
    CREATE TABLE IF NOT EXISTS transport_voucher_files (
        id SERIAL PRIMARY KEY,
        voucher_id INTEGER NOT NULL REFERENCES transport_vouchers(id) ON DELETE CASCADE,
        file_url VARCHAR(500) NOT NULL,
        filename VARCHAR(255) NOT NULL,
        mime_type VARCHAR(100),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # -- partner_invoice_transfers (association table)
    """
    CREATE TABLE IF NOT EXISTS partner_invoice_transfers (
        partner_invoice_id INTEGER NOT NULL REFERENCES partner_invoices(id) ON DELETE CASCADE,
        booking_transfer_id INTEGER NOT NULL REFERENCES booking_transfers(id) ON DELETE CASCADE,
        PRIMARY KEY (partner_invoice_id, booking_transfer_id)
    )
    """,
    # -- password_history
    """
    CREATE TABLE IF NOT EXISTS password_history (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        password_hash VARCHAR(255) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
]


# ===== INDEXES (IF NOT EXISTS) =====

CREATE_INDEXES = [
    # billing_parties
    "CREATE INDEX IF NOT EXISTS ix_billing_parties_company_id ON billing_parties(company_id)",
    "CREATE INDEX IF NOT EXISTS ix_billing_parties_company_type ON billing_parties(company_id, type)",
    # client_billing_parties
    "CREATE INDEX IF NOT EXISTS ix_client_billing_parties_client_id ON client_billing_parties(client_id)",
    "CREATE INDEX IF NOT EXISTS ix_client_billing_parties_billing_party_id ON client_billing_parties(billing_party_id)",
    "CREATE INDEX IF NOT EXISTS ix_client_billing_parties_client_default ON client_billing_parties(client_id, is_default)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_client_billing_parties_unique ON client_billing_parties(client_id, billing_party_id)",
    # billing_audit_logs
    "CREATE INDEX IF NOT EXISTS ix_billing_audit_logs_company_id ON billing_audit_logs(company_id)",
    "CREATE INDEX IF NOT EXISTS ix_billing_audit_logs_booking_id ON billing_audit_logs(booking_id)",
    "CREATE INDEX IF NOT EXISTS ix_billing_audit_logs_actor_user_id ON billing_audit_logs(actor_user_id)",
    # company_billing_profile
    "CREATE INDEX IF NOT EXISTS ix_company_billing_profile_company_id ON company_billing_profile(company_id)",
    "CREATE INDEX IF NOT EXISTS ix_company_billing_profile_uid_ide ON company_billing_profile(uid_ide)",
    # client_stays
    "CREATE INDEX IF NOT EXISTS ix_client_stays_client_id ON client_stays(client_id)",
    "CREATE INDEX IF NOT EXISTS ix_client_stays_company_id ON client_stays(company_id)",
    "CREATE INDEX IF NOT EXISTS ix_client_stays_created_by_user_id ON client_stays(created_by_user_id)",
    "CREATE INDEX IF NOT EXISTS ix_client_stays_client_start_date ON client_stays(client_id, start_date)",
    "CREATE INDEX IF NOT EXISTS ix_client_stays_company_start_date ON client_stays(company_id, start_date)",
    # clinic_billing_party_mappings
    "CREATE INDEX IF NOT EXISTS ix_clinic_billing_party_mappings_company_id ON clinic_billing_party_mappings(company_id)",
    "CREATE INDEX IF NOT EXISTS ix_clinic_billing_party_mappings_clinic_company_id ON clinic_billing_party_mappings(clinic_company_id)",
    "CREATE INDEX IF NOT EXISTS ix_clinic_billing_party_mappings_billing_party_id ON clinic_billing_party_mappings(billing_party_id)",
    # device_tokens
    "CREATE INDEX IF NOT EXISTS ix_device_tokens_driver_id ON device_tokens(driver_id)",
    "CREATE INDEX IF NOT EXISTS ix_device_tokens_token ON device_tokens(token)",
    "CREATE INDEX IF NOT EXISTS ix_device_tokens_driver_active ON device_tokens(driver_id, is_active)",
    # partner_invoices
    "CREATE INDEX IF NOT EXISTS ix_partner_invoices_partnership_id ON partner_invoices(partnership_id)",
    "CREATE INDEX IF NOT EXISTS ix_partner_invoices_executing_company_id ON partner_invoices(executing_company_id)",
    "CREATE INDEX IF NOT EXISTS ix_partner_invoices_period_year ON partner_invoices(period_year)",
    "CREATE INDEX IF NOT EXISTS ix_partner_invoices_period_month ON partner_invoices(period_month)",
    # transport_vouchers
    "CREATE INDEX IF NOT EXISTS ix_transport_vouchers_company_id ON transport_vouchers(company_id)",
    "CREATE INDEX IF NOT EXISTS ix_transport_vouchers_client_id ON transport_vouchers(client_id)",
    "CREATE INDEX IF NOT EXISTS ix_transport_vouchers_booking_id ON transport_vouchers(booking_id)",
    "CREATE INDEX IF NOT EXISTS ix_transport_vouchers_billing_party_id ON transport_vouchers(billing_party_id)",
    "CREATE INDEX IF NOT EXISTS ix_transport_vouchers_created_by_user_id ON transport_vouchers(created_by_user_id)",
    "CREATE INDEX IF NOT EXISTS ix_transport_vouchers_validated_by_user_id ON transport_vouchers(validated_by_user_id)",
    "CREATE INDEX IF NOT EXISTS ix_transport_vouchers_company_client_created ON transport_vouchers(company_id, client_id, created_at)",
    # transport_voucher_files
    "CREATE INDEX IF NOT EXISTS ix_transport_voucher_files_voucher_id ON transport_voucher_files(voucher_id)",
    # password_history
    "CREATE INDEX IF NOT EXISTS ix_password_history_user_id ON password_history(user_id)",
    "CREATE INDEX IF NOT EXISTS ix_password_history_created_at ON password_history(created_at)",
    "CREATE INDEX IF NOT EXISTS ix_password_history_user_created ON password_history(user_id, created_at)",
    # invoices - new columns indexes
    "CREATE INDEX IF NOT EXISTS ix_invoices_billing_party_id ON invoices(billing_party_id)",
    "CREATE INDEX IF NOT EXISTS ix_invoices_billed_to_company_id ON invoices(billed_to_company_id)",
]


# ===== COLUMNS on existing tables (ADD COLUMN IF NOT EXISTS) =====

ADD_COLUMNS = [
    # -- invoices : billing support
    "ALTER TABLE invoices ADD COLUMN IF NOT EXISTS billing_party_id INTEGER REFERENCES billing_parties(id) ON DELETE SET NULL",
    "ALTER TABLE invoices ADD COLUMN IF NOT EXISTS billing_strategy VARCHAR(50) NOT NULL DEFAULT 's1_patient'",
    "ALTER TABLE invoices ADD COLUMN IF NOT EXISTS billed_to_company_id INTEGER REFERENCES company(id) ON DELETE SET NULL",
    # -- company_billing_settings : SMTP + email signature
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS material_delivery_price_fixed NUMERIC(10,2)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS smtp_server VARCHAR(200)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS smtp_port INTEGER",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS smtp_use_tls BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS smtp_use_ssl BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS smtp_username VARCHAR(200)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS smtp_password VARCHAR(200)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS smtp_enabled BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS from_name VARCHAR(100)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS domain_verified BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS domain_dns_records JSON",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS email_signature_mode VARCHAR(10) NOT NULL DEFAULT 'form'",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS email_signature_text TEXT",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS signature_name VARCHAR(200)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS signature_title VARCHAR(200)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS signature_company VARCHAR(200)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS signature_phone_main VARCHAR(50)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS signature_phone_mobile VARCHAR(50)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS signature_email VARCHAR(200)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS signature_website VARCHAR(200)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS signature_address_line VARCHAR(200)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS signature_zip VARCHAR(10)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS signature_city VARCHAR(100)",
    "ALTER TABLE company_billing_settings ADD COLUMN IF NOT EXISTS email_signature_html_template TEXT",
    # -- invoice_reminders : rappels enrichis
    "ALTER TABLE invoice_reminders ADD COLUMN IF NOT EXISTS principal_amount NUMERIC(10,2) NOT NULL DEFAULT 0",
    "ALTER TABLE invoice_reminders ADD COLUMN IF NOT EXISTS reminder_fee_amount NUMERIC(10,2) NOT NULL DEFAULT 0",
    "ALTER TABLE invoice_reminders ADD COLUMN IF NOT EXISTS total_due NUMERIC(10,2) NOT NULL DEFAULT 0",
    "ALTER TABLE invoice_reminders ADD COLUMN IF NOT EXISTS qr_reference VARCHAR(50)",
    "ALTER TABLE invoice_reminders ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'OPEN'",
    "ALTER TABLE invoice_reminders ADD COLUMN IF NOT EXISTS paid_at TIMESTAMPTZ",
    # -- invoice_payments : lien rappel
    "ALTER TABLE invoice_payments ADD COLUMN IF NOT EXISTS reminder_id INTEGER REFERENCES invoice_reminders(id) ON DELETE SET NULL",
    # -- request_offers : enrichissement
    "ALTER TABLE request_offers ADD COLUMN IF NOT EXISTS mode VARCHAR(20) NOT NULL DEFAULT 'broadcast'",
    'ALTER TABLE request_offers ADD COLUMN IF NOT EXISTS "order" INTEGER NOT NULL DEFAULT 0',
    "ALTER TABLE request_offers ADD COLUMN IF NOT EXISTS sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    "ALTER TABLE request_offers ADD COLUMN IF NOT EXISTS rejection_reason TEXT",
    # -- transport_requests : champs detailles
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS mission_type VARCHAR(50) NOT NULL DEFAULT 'patient_transport'",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS delivery_description TEXT",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS scheduled_time TIMESTAMPTZ",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS pickup_location VARCHAR(255)",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS pickup_floor VARCHAR(50)",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS pickup_door_code VARCHAR(50)",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS dropoff_location VARCHAR(255)",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS dropoff_floor VARCHAR(50)",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS dropoff_door_code VARCHAR(50)",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS return_time TIMESTAMPTZ",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS mobility JSONB",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS floor_elevator_info TEXT",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS contact_on_site JSONB",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS accepted_at TIMESTAMPTZ",
    "ALTER TABLE transport_requests ADD COLUMN IF NOT EXISTS booking_id INTEGER REFERENCES booking(id) ON DELETE SET NULL",
    # -- institution_patients : champs additionnels
    "ALTER TABLE institution_patients ADD COLUMN IF NOT EXISTS dob DATE",
    "ALTER TABLE institution_patients ADD COLUMN IF NOT EXISTS gender VARCHAR(20)",
    "ALTER TABLE institution_patients ADD COLUMN IF NOT EXISTS notes TEXT",
    # -- institution_transport_preferences : order + updated_at
    'ALTER TABLE institution_transport_preferences ADD COLUMN IF NOT EXISTS "order" INTEGER NOT NULL DEFAULT 1',
    "ALTER TABLE institution_transport_preferences ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ",
]


def upgrade():
    # 1. Creer les tables manquantes
    for sql in CREATE_TABLES:
        op.execute(sql)

    # 2. Ajouter les colonnes manquantes sur les tables existantes
    for sql in ADD_COLUMNS:
        op.execute(sql)

    # 3. Creer les index manquants
    for sql in CREATE_INDEXES:
        op.execute(sql)


def downgrade():
    # Downgrade volontairement minimal : ne supprime que les tables
    # creees par cette migration. Les colonnes ajoutees sont conservees
    # pour eviter toute perte de donnees.
    tables_to_drop = [
        "partner_invoice_transfers",
        "transport_voucher_files",
        "transport_vouchers",
        "partner_invoices",
        "password_history",
        "device_tokens",
        "clinic_billing_party_mappings",
        "client_stays",
        "client_billing_parties",
        "company_billing_profile",
        "billing_audit_logs",
        "billing_parties",
    ]
    for table in tables_to_drop:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
