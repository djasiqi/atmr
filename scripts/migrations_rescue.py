#!/usr/bin/env python3
"""
===============================================================================
⚠️  OUTIL DE SECOURS - NE PAS UTILISER EN PREMIER RECOURS  ⚠️
===============================================================================

Ce script crée les tables manuellement SANS passer par Alembic.
Il est destiné uniquement aux situations où les migrations Alembic échouent
malgré l'utilisation de DISABLE_EVENTLET=1.

UTILISATION STANDARD (recommandée):
    DISABLE_EVENTLET=1 flask db upgrade heads

Si la méthode standard échoue, ALORS utiliser ce script:
    python scripts/migrations_rescue.py

Ce script:
- Contourne Alembic en créant les tables directement avec SQL
- Met à jour alembic_version pour marquer les migrations comme appliquées
- Utilise psycopg2 au lieu de psycopg3 pour éviter les problèmes avec eventlet

CONTEXTE:
Le problème vient du fait que eventlet.monkey_patch() (utilisé pour Socket.IO)
interfère avec les transactions Alembic/psycopg. psycopg3 utilise l'API async
native de libpq qui n'est pas compatible avec eventlet.

Voir docs/migrations.md pour plus de détails.
===============================================================================
"""

import os
import sys

# Vérification qu'on veut vraiment utiliser ce script
if os.getenv("I_KNOW_WHAT_I_AM_DOING") != "1":
    print("=" * 70)
    print("⚠️  ATTENTION: Ce script est un outil de secours uniquement!")
    print("=" * 70)
    print()
    print("Avez-vous essayé d'abord la méthode standard?")
    print("  DISABLE_EVENTLET=1 flask db upgrade heads")
    print()
    print("Si vous voulez vraiment utiliser ce script, relancez avec:")
    print("  I_KNOW_WHAT_I_AM_DOING=1 python scripts/migrations_rescue.py")
    print()
    sys.exit(1)

# URL avec psycopg2 (pas psycopg qui a des problèmes avec eventlet)
original_url = os.getenv("DATABASE_URL", "")
if not original_url:
    print("❌ DATABASE_URL non définie!")
    sys.exit(1)

if "+psycopg://" in original_url and "+psycopg2://" not in original_url:
    db_url = original_url.replace("+psycopg://", "+psycopg2://")
else:
    db_url = original_url

print(f"📊 Database URL: {db_url.split('@')[0]}@***")

from sqlalchemy import create_engine, text, pool

engine = create_engine(db_url, poolclass=pool.NullPool)


def execute_with_commit(sql_statements: list[str], description: str = "") -> bool:
    """Exécute une liste de statements SQL avec commit explicite."""
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            for sql in sql_statements:
                conn.execute(text(sql))
            trans.commit()
            if description:
                print(f"✅ {description}")
            return True
        except Exception as e:
            trans.rollback()
            print(f"❌ Erreur {description}: {e}")
            return False


# Vérifier la version actuelle
print("\n=== Vérification de l'état actuel ===")
with engine.connect() as conn:
    result = conn.execute(text("SELECT version_num FROM alembic_version"))
    current = result.scalar()
    print(f"Version actuelle: {current}")

    # Vérifier si institutions existe
    result = conn.execute(
        text(
            """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = 'institutions'
        )
    """
        )
    )
    has_institutions = result.scalar()
    print(f"Table institutions existe: {has_institutions}")

if has_institutions:
    print("\n✅ La table institutions existe déjà - migrations probablement à jour!")
    print("Mise à jour de alembic_version vers head...")
    execute_with_commit(
        ["UPDATE alembic_version SET version_num = '20260204_audit_immut'"],
        "alembic_version mis à jour",
    )
    sys.exit(0)

# Créer les tables institution manuellement
print("\n=== Création manuelle des tables institution ===")

# Table institutions
institutions_sql = """
CREATE TABLE IF NOT EXISTS institutions (
    id SERIAL PRIMARY KEY,
    public_id VARCHAR(36) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    legal_name VARCHAR(200),
    gln_number VARCHAR(20),
    zsr_number VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    postal_code VARCHAR(20),
    canton VARCHAR(2),
    country VARCHAR(2) DEFAULT 'CH',
    contact_email VARCHAR(254),
    contact_phone VARCHAR(20),
    billing_email VARCHAR(254),
    settings_json JSON,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# Table institution_users
institution_users_sql = """
CREATE TABLE IF NOT EXISTS institution_users (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES "user"(id) ON DELETE CASCADE,
    institution_id INTEGER REFERENCES institutions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'institution_requester',
    department VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, institution_id)
)
"""

# Table institution_api_keys
api_keys_sql = """
CREATE TABLE IF NOT EXISTS institution_api_keys (
    id SERIAL PRIMARY KEY,
    public_id VARCHAR(36) NOT NULL UNIQUE,
    institution_id INTEGER NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    key_prefix VARCHAR(8) NOT NULL,
    key_hash VARCHAR(64) NOT NULL,
    scopes TEXT[] DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMP,
    created_by_user_id INTEGER REFERENCES "user"(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP
)
"""

# Table institution_patients
patients_sql = """
CREATE TABLE IF NOT EXISTS institution_patients (
    id SERIAL PRIMARY KEY,
    public_id VARCHAR(36) NOT NULL UNIQUE,
    institution_id INTEGER NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
    external_reference VARCHAR(100),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    birth_date DATE,
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    postal_code VARCHAR(20),
    mobility_notes TEXT,
    medical_notes TEXT,
    avs_number VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (institution_id, external_reference)
)
"""

# Table transport_requests
transport_requests_sql = """
DO $$ BEGIN
    CREATE TYPE transport_request_status AS ENUM (
        'DRAFT', 'SENT', 'ACCEPTED', 'CONVERTED', 'CANCELLED', 'EXPIRED'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS transport_requests (
    id SERIAL PRIMARY KEY,
    public_id VARCHAR(36) NOT NULL UNIQUE,
    institution_id INTEGER NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
    patient_id INTEGER REFERENCES institution_patients(id),
    external_reference VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT',
    scheduled_datetime TIMESTAMP NOT NULL,
    pickup_address TEXT NOT NULL,
    pickup_lat FLOAT,
    pickup_lng FLOAT,
    dropoff_address TEXT NOT NULL,
    dropoff_lat FLOAT,
    dropoff_lng FLOAT,
    is_round_trip BOOLEAN DEFAULT FALSE,
    return_datetime TIMESTAMP,
    mobility_requirements TEXT[] DEFAULT '{}',
    special_equipment TEXT[] DEFAULT '{}',
    notes TEXT,
    billing_intent VARCHAR(50),
    billing_details JSON,
    accepted_by_company_id INTEGER REFERENCES company(id),
    booking_id INTEGER,
    created_by_user_id INTEGER REFERENCES "user"(id),
    sent_at TIMESTAMP,
    converted_at TIMESTAMP,
    cancelled_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# Table request_offers
request_offers_sql = """
DO $$ BEGIN
    CREATE TYPE request_offer_status AS ENUM (
        'PENDING', 'ACCEPTED', 'DECLINED', 'EXPIRED', 'UNAVAILABLE'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS request_offers (
    id SERIAL PRIMARY KEY,
    public_id VARCHAR(36) NOT NULL UNIQUE,
    transport_request_id INTEGER NOT NULL REFERENCES transport_requests(id) ON DELETE CASCADE,
    company_id INTEGER NOT NULL REFERENCES company(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    priority_rank INTEGER NOT NULL DEFAULT 0,
    is_global_fallback BOOLEAN DEFAULT FALSE,
    offered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    responded_at TIMESTAMP,
    decline_reason TEXT
)
"""

# Table institution_transport_preferences
preferences_sql = """
CREATE TABLE IF NOT EXISTS institution_transport_preferences (
    id SERIAL PRIMARY KEY,
    institution_id INTEGER NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
    company_id INTEGER NOT NULL REFERENCES company(id) ON DELETE CASCADE,
    priority_rank INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (institution_id, company_id)
)
"""

# Exécuter toutes les créations de tables
success = True
success = (
    execute_with_commit([institutions_sql], "Table institutions créée") and success
)
success = (
    execute_with_commit([institution_users_sql], "Table institution_users créée")
    and success
)
success = (
    execute_with_commit([api_keys_sql], "Table institution_api_keys créée") and success
)
success = (
    execute_with_commit([patients_sql], "Table institution_patients créée") and success
)
success = (
    execute_with_commit([transport_requests_sql], "Table transport_requests créée")
    and success
)
success = (
    execute_with_commit([request_offers_sql], "Table request_offers créée") and success
)
success = (
    execute_with_commit(
        [preferences_sql], "Table institution_transport_preferences créée"
    )
    and success
)

# Mettre à jour alembic_version
if success:
    execute_with_commit(
        ["UPDATE alembic_version SET version_num = '20260204_audit_immut'"],
        "Version alembic mise à jour vers head",
    )

# Vérification finale
print("\n=== Vérification finale ===")
with engine.connect() as conn:
    result = conn.execute(text("SELECT version_num FROM alembic_version"))
    print(f"Version: {result.scalar()}")

    result = conn.execute(
        text(
            """
        SELECT table_name FROM information_schema.tables 
        WHERE table_name IN ('institutions', 'transport_requests', 'request_offers')
        ORDER BY table_name
    """
        )
    )
    tables = [row[0] for row in result.fetchall()]
    print(f"Tables créées: {tables}")

print("\n✅ Script terminé!")
