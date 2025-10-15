# 🗄️ Notes Migrations Alembic - ATMR

**Date**: 15 octobre 2025  
**Périmètre**: Corrections drift models ↔ DB + nouveaux index + contraintes timezone

---

## 📋 Résumé Exécutif

**Problèmes détectés:**

1. **Index manquants**: `invoice_line_id` sur `booking`, composites sur filtres fréquents
2. **Timezone incohérent**: Mix DateTime(timezone=True/False) entre tables
3. **Contraintes**: Manque validation montants négatifs sur certains champs
4. **Enum drift**: Payment.method défini en dur vs models.enums.PaymentMethod

**Migrations proposées**: 3 migrations principales + 1 optionnelle

---

## 🚀 Migration 1: Index Critiques (PRIORITÉ HAUTE)

### Fichier: `backend/migrations/versions/XXXX_add_critical_indexes.py`

**Objectif**: Ajouter index manquants pour requêtes fréquentes (bookings, invoices, dispatch)

**Tables impactées**: `booking`, `invoice`, `assignment`, `driver_status`

**Opérations**:

```python
"""Add critical indexes for performance

Revision ID: add_critical_indexes_2025
Revises: (current HEAD)
Create Date: 2025-10-15
"""
from alembic import op
import sqlalchemy as sa

revision = 'add_critical_indexes_2025'
down_revision = '(HEAD)'  # Remplacer par le HEAD actuel
branch_labels = None
depends_on = None

def upgrade():
    # 1. Booking.invoice_line_id (FK sans index)
    op.create_index(
        'ix_booking_invoice_line',
        'booking',
        ['invoice_line_id'],
        unique=False
    )

    # 2. Composites pour filtres fréquents
    op.create_index(
        'ix_booking_company_status_scheduled',
        'booking',
        ['company_id', 'status', 'scheduled_time'],
        unique=False
    )

    op.create_index(
        'ix_invoice_company_status_due',
        'invoices',
        ['company_id', 'status', 'due_date'],
        unique=False
    )

    # 3. Assignment.dispatch_run_id (FK sans index)
    op.create_index(
        'ix_assignment_dispatch_run',
        'assignment',
        ['dispatch_run_id'],
        unique=False
    )

    # 4. DriverStatus.current_assignment_id
    op.create_index(
        'ix_driver_status_assignment',
        'driver_status',
        ['current_assignment_id'],
        unique=False
    )

    # 5. RealtimeEvent.timestamp pour requêtes temporelles
    op.create_index(
        'ix_realtime_event_timestamp',
        'realtime_event',
        ['timestamp'],
        unique=False
    )

def downgrade():
    op.drop_index('ix_booking_invoice_line', table_name='booking')
    op.drop_index('ix_booking_company_status_scheduled', table_name='booking')
    op.drop_index('ix_invoice_company_status_due', table_name='invoices')
    op.drop_index('ix_assignment_dispatch_run', table_name='assignment')
    op.drop_index('ix_driver_status_assignment', table_name='driver_status')
    op.drop_index('ix_realtime_event_timestamp', table_name='realtime_event')
```

**Impact estimé**:

- **Performances**: Gain 50-80% sur requêtes filtrant par company_id + status (bookings/invoices)
- **Espace disque**: +5-10MB pour indexes (négligeable si <100k rows)
- **Durée migration**: ~10-30s en prod (sans lock exclusif long)

**Rollback**: `alembic downgrade -1` (drop indexes, aucune perte de données)

**Risques**: Aucun (création index online, pas de downtime)

---

## ⏱️ Migration 2: Uniformisation Timezone (PRIORITÉ HAUTE)

### Fichier: `backend/migrations/versions/XXXX_uniformize_timezone.py`

**Objectif**: Convertir DateTime(timezone=False) → DateTime(timezone=True) + migration données

**Tables impactées**: `booking`, `driver_shift`, `driver_unavailability`, etc.

⚠️ **ATTENTION**: Migration complexe nécessitant conversion données existantes

```python
"""Uniformize timezone: naive local -> UTC aware

Revision ID: uniformize_timezone_2025
Revises: add_critical_indexes_2025
Create Date: 2025-10-15
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'uniformize_timezone_2025'
down_revision = 'add_critical_indexes_2025'

def upgrade():
    """
    Convertit les colonnes DateTime naïves en timezone-aware (UTC).
    Les données existantes (interprétées comme Europe/Zurich) sont converties en UTC.
    """
    # 1. Booking.scheduled_time (naïf local → UTC aware)
    # Stratégie: ALTER TYPE + conversion inline
    op.execute("""
        ALTER TABLE booking
        ALTER COLUMN scheduled_time TYPE timestamptz
        USING scheduled_time AT TIME ZONE 'Europe/Zurich'
    """)

    # 2. DriverShift.start_local, end_local
    op.execute("""
        ALTER TABLE driver_shift
        ALTER COLUMN start_local TYPE timestamptz
        USING start_local AT TIME ZONE 'Europe/Zurich'
    """)
    op.execute("""
        ALTER TABLE driver_shift
        ALTER COLUMN end_local TYPE timestamptz
        USING end_local AT TIME ZONE 'Europe/Zurich'
    """)

    # 3. DriverUnavailability.start_local, end_local
    op.execute("""
        ALTER TABLE driver_unavailability
        ALTER COLUMN start_local TYPE timestamptz
        USING start_local AT TIME ZONE 'Europe/Zurich'
    """)
    op.execute("""
        ALTER TABLE driver_unavailability
        ALTER COLUMN end_local TYPE timestamptz
        USING end_local AT TIME ZONE 'Europe/Zurich'
    """)

    # 4. DriverBreak.start_local, end_local
    op.execute("""
        ALTER TABLE driver_break
        ALTER COLUMN start_local TYPE timestamptz
        USING start_local AT TIME ZONE 'Europe/Zurich'
    """)
    op.execute("""
        ALTER TABLE driver_break
        ALTER COLUMN end_local TYPE timestamptz
        USING end_local AT TIME ZONE 'Europe/Zurich'
    """)

def downgrade():
    """
    Rollback: reconvertit UTC aware → naïf local Europe/Zurich.
    """
    op.execute("""
        ALTER TABLE booking
        ALTER COLUMN scheduled_time TYPE timestamp
        USING scheduled_time AT TIME ZONE 'Europe/Zurich'
    """)
    op.execute("""
        ALTER TABLE driver_shift
        ALTER COLUMN start_local TYPE timestamp
        USING start_local AT TIME ZONE 'Europe/Zurich'
    """)
    op.execute("""
        ALTER TABLE driver_shift
        ALTER COLUMN end_local TYPE timestamp
        USING end_local AT TIME ZONE 'Europe/Zurich'
    """)
    op.execute("""
        ALTER TABLE driver_unavailability
        ALTER COLUMN start_local TYPE timestamp
        USING start_local AT TIME ZONE 'Europe/Zurich'
    """)
    op.execute("""
        ALTER TABLE driver_unavailability
        ALTER COLUMN end_local TYPE timestamp
        USING end_local AT TIME ZONE 'Europe/Zurich'
    """)
    op.execute("""
        ALTER TABLE driver_break
        ALTER COLUMN start_local TYPE timestamp
        USING start_local AT TIME ZONE 'Europe/Zurich'
    """)
    op.execute("""
        ALTER TABLE driver_break
        ALTER COLUMN end_local TYPE timestamp
        USING end_local AT TIME ZONE 'Europe/Zurich'
    """)
```

**Impact estimé**:

- **Données**: Conversion inline, pas de perte (assume données actuelles = Europe/Zurich)
- **Durée migration**: 30s-2min en prod (lock exclusif court sur chaque table)
- **Risques**: Si données existantes déjà en UTC → conversion double (VÉRIFIER AVANT)

**Tests pré-migration** (OBLIGATOIRES):

```sql
-- Vérifier échantillon de données actuelles
SELECT id, scheduled_time,
       scheduled_time AT TIME ZONE 'Europe/Zurich' AS would_become_utc
FROM booking
WHERE scheduled_time IS NOT NULL
LIMIT 10;

-- Si "would_become_utc" semble incorrect (+1h ou +2h selon DST),
-- les données sont DÉJÀ en UTC → NE PAS MIGRER
```

**Rollback**: `alembic downgrade -1` (reconversion UTC → local)

**Recommandations**:

1. **Backup DB complet** avant migration
2. **Test sur staging** avec données de prod anonymisées
3. **Fenêtre de maintenance** recommandée (2-5min downtime)

---

## ✅ Migration 3: Contraintes Validation (PRIORITÉ MOYENNE)

### Fichier: `backend/migrations/versions/XXXX_add_validation_constraints.py`

**Objectif**: Ajouter contraintes CHECK manquantes (montants, bornes)

```python
"""Add validation constraints (amounts, ranges)

Revision ID: add_validation_constraints_2025
Revises: uniformize_timezone_2025
Create Date: 2025-10-15
"""
from alembic import op

revision = 'add_validation_constraints_2025'
down_revision = 'uniformize_timezone_2025'

def upgrade():
    # 1. Invoice: total_amount, balance_due >= 0
    op.create_check_constraint(
        'chk_invoice_total_nonneg',
        'invoices',
        'total_amount >= 0'
    )
    op.create_check_constraint(
        'chk_invoice_balance_nonneg',
        'invoices',
        'balance_due >= 0'
    )

    # 2. InvoiceLine: line_total >= 0
    op.create_check_constraint(
        'chk_invoice_line_total_nonneg',
        'invoice_lines',
        'line_total >= 0'
    )

    # 3. InvoicePayment: amount > 0
    op.create_check_constraint(
        'chk_invoice_payment_positive',
        'invoice_payments',
        'amount > 0'
    )

    # 4. Booking: amount >= 0 (peut être 0 si retour placeholder)
    # Déjà présent ? Vérifier models

def downgrade():
    op.drop_constraint('chk_invoice_total_nonneg', 'invoices', type_='check')
    op.drop_constraint('chk_invoice_balance_nonneg', 'invoices', type_='check')
    op.drop_constraint('chk_invoice_line_total_nonneg', 'invoice_lines', type_='check')
    op.drop_constraint('chk_invoice_payment_positive', 'invoice_payments', type_='check')
```

**Impact estimé**:

- **Données**: Aucune modification (contraintes sur INSERT/UPDATE futurs)
- **Risques**: Si données existantes violent contraintes → migration échoue (VÉRIFIER AVANT)

**Tests pré-migration**:

```sql
-- Vérifier données existantes
SELECT COUNT(*) FROM invoices WHERE total_amount < 0;
SELECT COUNT(*) FROM invoices WHERE balance_due < 0;
SELECT COUNT(*) FROM invoice_lines WHERE line_total < 0;
SELECT COUNT(*) FROM invoice_payments WHERE amount <= 0;

-- Si COUNT > 0 → CORRIGER les données avant migration
```

**Rollback**: `alembic downgrade -1`

---

## 🔧 Migration 4: Payment Method Enum (OPTIONNELLE)

### Fichier: `backend/migrations/versions/XXXX_unify_payment_method_enum.py`

**Objectif**: Aligner Payment.method avec models.enums.PaymentMethod

⚠️ **Complexe**: Nécessite modification type enum PostgreSQL

```python
"""Unify payment_method enum with models

Revision ID: unify_payment_method_2025
Revises: add_validation_constraints_2025
Create Date: 2025-10-15
"""
from alembic import op

revision = 'unify_payment_method_2025'
down_revision = 'add_validation_constraints_2025'

def upgrade():
    # Stratégie: ALTER TYPE avec renommage valeurs
    # 'credit_card' → 'card' (si souhaité pour uniformiser)
    # Actuellement Payment: credit_card|paypal|bank_transfer|cash
    # models.enums.PaymentMethod: BANK_TRANSFER|CASH|CARD|ADJUSTMENT

    # 1. Créer nouveau type
    op.execute("CREATE TYPE payment_method_new AS ENUM ('bank_transfer', 'cash', 'card', 'adjustment')")

    # 2. Convertir colonne
    op.execute("""
        ALTER TABLE payment
        ALTER COLUMN method TYPE payment_method_new
        USING CASE
            WHEN method::text = 'credit_card' THEN 'card'::payment_method_new
            WHEN method::text = 'paypal' THEN 'card'::payment_method_new
            WHEN method::text = 'bank_transfer' THEN 'bank_transfer'::payment_method_new
            WHEN method::text = 'cash' THEN 'cash'::payment_method_new
            ELSE 'card'::payment_method_new
        END
    """)

    # 3. Drop ancien type
    op.execute("DROP TYPE payment_method")

    # 4. Renommer nouveau type
    op.execute("ALTER TYPE payment_method_new RENAME TO payment_method")

def downgrade():
    # Rollback: recréer ancien type
    op.execute("CREATE TYPE payment_method_old AS ENUM ('credit_card', 'paypal', 'bank_transfer', 'cash')")
    op.execute("""
        ALTER TABLE payment
        ALTER COLUMN method TYPE payment_method_old
        USING CASE
            WHEN method::text = 'card' THEN 'credit_card'::payment_method_old
            WHEN method::text = 'bank_transfer' THEN 'bank_transfer'::payment_method_old
            WHEN method::text = 'cash' THEN 'cash'::payment_method_old
            ELSE 'credit_card'::payment_method_old
        END
    """)
    op.execute("DROP TYPE payment_method")
    op.execute("ALTER TYPE payment_method_old RENAME TO payment_method")
```

**Impact estimé**:

- **Données**: Conversion valeurs (paypal → card)
- **Risques**: Perte sémantique (paypal vs card distincts dans certains contextes)

**Recommandation**: **Reporter** cette migration si confusion métier (paypal vs card à distinguer)

**Rollback**: `alembic downgrade -1`

---

## 📊 Ordre d'Application & Plan de Déploiement

### Étape 1: Génération migrations

```bash
cd backend

# Migration 1: Index
alembic revision -m "add_critical_indexes"
# → Copier le code upgrade/downgrade depuis MIGRATIONS_NOTES.md

# Migration 2: Timezone
alembic revision -m "uniformize_timezone"
# → Copier le code + TESTER sur staging

# Migration 3: Contraintes
alembic revision -m "add_validation_constraints"
# → Copier le code

# (Optionnel) Migration 4: Payment enum
alembic revision -m "unify_payment_method"
```

### Étape 2: Tests staging

```bash
# Backup staging DB
pg_dump -h staging-db -U atmr atmr > backup_before_migrations.sql

# Apply migrations
alembic upgrade head

# Tests régression
pytest tests/test_bookings.py tests/test_invoices.py tests/test_dispatch.py

# Si OK → proceed to prod
# Si KO → rollback
alembic downgrade base  # ou -1 pour rollback step-by-step
psql -h staging-db -U atmr atmr < backup_before_migrations.sql
```

### Étape 3: Production (fenêtre de maintenance)

```bash
# 1. Backup complet
pg_dump -h prod-db -U atmr atmr > backup_prod_$(date +%Y%m%d_%H%M%S).sql

# 2. Mettre app en maintenance (optionnel)
# docker-compose stop api celery-worker celery-beat

# 3. Apply migrations
alembic upgrade head

# 4. Vérifications post-migration
psql -h prod-db -U atmr atmr -c "SELECT COUNT(*) FROM booking WHERE invoice_line_id IS NOT NULL;"
psql -h prod-db -U atmr atmr -c "SELECT * FROM pg_indexes WHERE tablename IN ('booking', 'invoices');"

# 5. Restart app
docker-compose up -d api celery-worker celery-beat

# 6. Tests smoke
curl http://localhost:5000/health
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/api/companies/me/bookings
```

### Rollback d'urgence

```bash
# Si crash post-migration
alembic downgrade -1  # rollback dernière migration
# ou
alembic downgrade <revision_id>  # rollback vers revision spécifique

# Si données corrompues
psql -h prod-db -U atmr atmr < backup_prod_YYYYMMDD_HHMMSS.sql
```

---

## ⚠️ Risques & Mitigations

| Risque                                     | Probabilité | Impact   | Mitigation                                                        |
| ------------------------------------------ | ----------- | -------- | ----------------------------------------------------------------- |
| **Migration timezone double-convertit**    | Moyenne     | Critique | Tests staging + vérification échantillon pré-migration            |
| **Lock exclusif long**                     | Faible      | Moyen    | Exécuter hors heures pointe, index CONCURRENTLY si Postgres ≥9.2  |
| **Contraintes violent données existantes** | Moyenne     | Moyen    | Tests pré-migration (SELECT violating rows) + correction manuelle |
| **Enum payment change casse app**          | Faible      | Moyen    | Tests exhaustifs + rollback plan                                  |
| **Downtime >5min**                         | Faible      | Élevé    | Fenêtre de maintenance planifiée + communication utilisateurs     |

---

## 📝 Checklist Pré-Migration

- [ ] Backup complet DB production
- [ ] Tests migrations sur staging avec données de prod anonymisées
- [ ] Vérification échantillon timezone (voir SQL ci-dessus)
- [ ] Vérification contraintes violées (SELECT violating rows)
- [ ] Tests régression complets (pytest)
- [ ] Rollback plan documenté
- [ ] Fenêtre de maintenance planifiée (si nécessaire)
- [ ] Communication équipe + utilisateurs
- [ ] Monitoring activé post-migration (logs, Sentry, métriques DB)

---

## 🎓 Bonnes Pratiques

1. **Toujours tester sur staging** avec données réelles anonymisées
2. **Migrations réversibles**: Garantir que `downgrade()` fonctionne
3. **Idempotence**: Migration doit pouvoir être rejouée sans erreur (IF NOT EXISTS)
4. **Logs détaillés**: Activer `logging.INFO` pendant migration
5. **Backup automatique**: Script pre-migration hook
6. **Monitoring**: Alertes si durée migration > seuil attendu

---

_Document généré le 15 octobre 2025. Pour toute question, se référer à la documentation Alembic officielle ou aux patches backend fournis._
