# 🖥️ COMMANDES SEMAINE 2

**Toutes les commandes prêtes à copier-coller.**

---

## 🔧 SETUP INITIAL

### Créer dossiers

```bash
# Dossier backup DB
mkdir -p session/backup_semaine2

# Dossier profiling
mkdir -p backend/scripts/profiling

# Vérifier
ls -la session/backup_semaine2/
```

### Backup Base de Données

```bash
cd backend

# SQLite
cp instance/development.db ../session/backup_semaine2/development.db.backup

# PostgreSQL
pg_dump -U postgres -d atmr_db > ../session/backup_semaine2/db_backup.sql

# Vérifier backup
ls -la ../session/backup_semaine2/
```

### Installer packages

```bash
# Profiling packages
.\venv\Scripts\python.exe -m pip install nplusone
.\venv\Scripts\python.exe -m pip install sqlalchemy-utils
.\venv\Scripts\python.exe -m pip install matplotlib

# Ajouter à requirements
echo "nplusone==1.0.0" >> requirements.txt
echo "matplotlib==3.8.0" >> requirements.txt
```

---

## 📅 JOUR 1 : PROFILING

### Activer SQL logging

```bash
cd backend

# Créer .env.profiling
cat > .env.profiling << 'EOF'
SQL_ECHO=true
FLASK_DEBUG=true
EOF

# Lancer avec profiling
set FLASK_ENV=profiling
.\venv\Scripts\python.exe app.py
```

### Profiling script

```bash
# Créer dossier
mkdir -p scripts/profiling

# Créer script (contenu dans guide)
touch scripts/profiling/profile_dispatch.py

# Lancer profiling
.\venv\Scripts\python.exe scripts/profiling/profile_dispatch.py
```

### Analyser queries

```bash
# Voir toutes les queries
tail -f logs/app.log | grep "SELECT"

# Compter queries
.\venv\Scripts\python.exe -c "
import re
with open('logs/app.log') as f:
    queries = [line for line in f if 'SELECT' in line]
    print(f'Total queries: {len(queries)}')
"
```

### Voir index existants

```bash
# SQLite
.\venv\Scripts\python.exe -c "
from ext import db
result = db.engine.execute('SELECT name FROM sqlite_master WHERE type=\"index\"')
for row in result:
    print(row[0])
"

# PostgreSQL
.\venv\Scripts\python.exe -c "
from ext import db
result = db.engine.execute('SELECT indexname FROM pg_indexes WHERE schemaname=\"public\"')
for row in result:
    print(row[0])
"
```

---

## 📅 JOUR 2 : INDEX DB

### Créer migration

```bash
cd backend

# Créer migration
.\venv\Scripts\python.exe -m flask db revision -m "add_performance_indexes"

# Fichier créé
ls -la migrations/versions/*_add_performance_indexes.py
```

### Appliquer migration

```bash
# Upgrade (appliquer)
.\venv\Scripts\python.exe -m flask db upgrade

# Vérifier version
.\venv\Scripts\python.exe -m flask db current

# Voir historique
.\venv\Scripts\python.exe -m flask db history
```

### Tester downgrade

```bash
# Downgrade (retirer index)
.\venv\Scripts\python.exe -m flask db downgrade

# Re-upgrade
.\venv\Scripts\python.exe -m flask db upgrade
```

### Vérifier index créés

```bash
# SQLite - Voir index d'une table
.\venv\Scripts\python.exe -c "
from ext import db
indexes = db.engine.execute('PRAGMA index_list(assignment)').fetchall()
print('Index sur assignment:')
for idx in indexes:
    print(f'  - {idx[1]}')
"

# PostgreSQL - Voir index
.\venv\Scripts\python.exe -c "
from ext import db
result = db.engine.execute('''
    SELECT indexname, indexdef
    FROM pg_indexes
    WHERE tablename = 'assignment'
''')
for row in result:
    print(f'{row[0]}: {row[1]}')
"
```

### Benchmark avant/après

```bash
# AVANT index
.\venv\Scripts\python.exe -m flask db downgrade
.\venv\Scripts\python.exe scripts/profiling/benchmark_indexes.py
# Noter: Temps = ___s

# APRÈS index
.\venv\Scripts\python.exe -m flask db upgrade
.\venv\Scripts\python.exe scripts/profiling/benchmark_indexes.py
# Noter: Temps = ___s

# Calculer gain
.\venv\Scripts\python.exe -c "
avant = float(input('Temps avant: '))
apres = float(input('Temps après: '))
gain = ((avant - apres) / avant) * 100
print(f'Gain: {gain:.1f}%')
"
```

---

## 📅 JOUR 3 : BULK INSERTS

### Backup apply.py

```bash
cd backend/services/unified_dispatch
cp apply.py apply.py.backup

# Vérifier backup
ls -la apply.py*
```

### Trouver commits multiples

```bash
# Chercher tous les commits
grep -n "\.commit()" apply.py

# Chercher boucles avec commit
grep -B 5 "\.commit()" apply.py | grep "for "
```

### Tester bulk insert

```bash
# Test rapide dans Python shell
.\venv\Scripts\python.exe

>>> from ext import db
>>> from models import Assignment
>>>
>>> # Bulk insert test
>>> assignments = [
...     {'booking_id': 1, 'driver_id': 10, 'status': 'pending'},
...     {'booking_id': 2, 'driver_id': 11, 'status': 'pending'},
... ]
>>>
>>> db.session.bulk_insert_mappings(Assignment, assignments)
>>> db.session.commit()
>>> print("✅ Bulk insert OK")
>>> exit()
```

### Benchmark bulk vs loop

```bash
# Créer script benchmark
cat > scripts/profiling/benchmark_bulk.py << 'EOF'
import time
from ext import db
from models import Assignment

# Test loop (old way)
start = time.time()
for i in range(100):
    a = Assignment(booking_id=i, driver_id=i, status='test')
    db.session.add(a)
    db.session.commit()
end = time.time()
print(f"Loop + commits: {end-start:.2f}s")

# Clean
Assignment.query.filter_by(status='test').delete()
db.session.commit()

# Test bulk (new way)
start = time.time()
assignments = [{'booking_id': i, 'driver_id': i, 'status': 'test'} for i in range(100)]
db.session.bulk_insert_mappings(Assignment, assignments)
db.session.commit()
end = time.time()
print(f"Bulk insert: {end-start:.2f}s")

# Clean
Assignment.query.filter_by(status='test').delete()
db.session.commit()
EOF

.\venv\Scripts\python.exe scripts/profiling/benchmark_bulk.py
```

---

## 📅 JOUR 4 : QUERIES N+1

### Installer nplusone

```bash
.\venv\Scripts\python.exe -m pip install nplusone
```

### Activer détection

```python
# Dans app.py
from nplusone.ext.flask_sqlalchemy import NPlusOne

def create_app():
    app = Flask(__name__)
    # ... config ...

    # Activer N+1 detection
    NPlusOne(app)

    return app
```

### Détecter N+1

```bash
# Lancer app
.\venv\Scripts\python.exe app.py

# Faire un dispatch
curl -X POST http://localhost:5000/api/dispatch/run \
  -H "Content-Type: application/json" \
  -d '{"company_id": 1, "for_date": "2025-10-21"}'

# Voir logs
tail -100 logs/app.log | grep "NPlusOne"
```

### Corriger avec joinedload

```bash
# Exemple de correction
# AVANT
# bookings = Booking.query.all()

# APRÈS
# from sqlalchemy.orm import joinedload
# bookings = Booking.query.options(joinedload(Booking.client)).all()
```

---

## 📅 JOUR 5 : VALIDATION

### Benchmark complet

```bash
# Lancer benchmark final
.\venv\Scripts\python.exe scripts/profiling/final_benchmark.py

# Sauvegarder résultats
.\venv\Scripts\python.exe scripts/profiling/final_benchmark.py > scripts/profiling/final_results.txt
```

### Tests tous les modules

```bash
# Tous les tests
.\venv\Scripts\python.exe -m pytest tests/ -v

# Tests spécifiques nouveaux
.\venv\Scripts\python.exe -m pytest tests/test_apply_bulk.py -v

# Coverage
.\venv\Scripts\python.exe -m pytest tests/ --cov=backend --cov-report=html
```

### Vérifier performance

```bash
# Temps dispatch
.\venv\Scripts\python.exe -c "
from services.unified_dispatch.engine import DispatchEngine
import time

start = time.time()
engine = DispatchEngine()
result = engine.run(company_id=1, for_date='2025-10-21')
end = time.time()

print(f'Temps: {end-start:.2f}s')
print(f'Assignments: {len(result[\"assignments\"])}')
"
```

### Créer rapport final

```bash
touch scripts/profiling/PERFORMANCE_REPORT.md
code scripts/profiling/PERFORMANCE_REPORT.md
# Remplir avec résultats
```

---

## 🚨 COMMANDES URGENCES

### Rollback Migration

```bash
cd backend

# Voir version actuelle
.\venv\Scripts\python.exe -m flask db current

# Downgrade d'une version
.\venv\Scripts\python.exe -m flask db downgrade

# Downgrade à version spécifique
.\venv\Scripts\python.exe -m flask db downgrade <revision_id>
```

### Restaurer DB depuis backup

```bash
# SQLite
cp session/backup_semaine2/development.db.backup backend/instance/development.db

# PostgreSQL
psql -U postgres -d atmr_db < session/backup_semaine2/db_backup.sql
```

### Supprimer tous les index (si problème)

```bash
# Créer script cleanup
cat > scripts/profiling/cleanup_indexes.py << 'EOF'
from ext import db

# Liste index à supprimer
indexes = [
    'idx_assignment_booking_created',
    'idx_booking_status_scheduled_company',
    'idx_driver_company_available',
    'idx_booking_company_scheduled',
    'idx_assignment_run_status',
]

for idx in indexes:
    try:
        db.engine.execute(f'DROP INDEX IF EXISTS {idx}')
        print(f'✅ Supprimé {idx}')
    except Exception as e:
        print(f'❌ Erreur {idx}: {e}')

db.session.commit()
EOF

.\venv\Scripts\python.exe scripts/profiling/cleanup_indexes.py
```

---

## 📝 COMMANDES UTILES

### Voir schéma table

```bash
# SQLite
.\venv\Scripts\python.exe -c "
from ext import db
schema = db.engine.execute('PRAGMA table_info(assignment)').fetchall()
for col in schema:
    print(f'{col[1]} {col[2]}')
"

# PostgreSQL
.\venv\Scripts\python.exe -c "
from ext import db
result = db.engine.execute('''
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'assignment'
''')
for row in result:
    print(f'{row[0]}: {row[1]}')
"
```

### Analyser query plan

```bash
# SQLite
.\venv\Scripts\python.exe -c "
from ext import db
plan = db.engine.execute('EXPLAIN QUERY PLAN SELECT * FROM assignment WHERE booking_id = 1').fetchall()
for row in plan:
    print(row)
"

# PostgreSQL
.\venv\Scripts\python.exe -c "
from ext import db
plan = db.engine.execute('EXPLAIN ANALYZE SELECT * FROM assignment WHERE booking_id = 1').fetchall()
for row in plan:
    print(row[0])
"
```

### Stats DB

```bash
# Nombre de lignes par table
.\venv\Scripts\python.exe -c "
from ext import db
from models import Assignment, Booking, Driver

print(f'Assignments: {Assignment.query.count()}')
print(f'Bookings: {Booking.query.count()}')
print(f'Drivers: {Driver.query.count()}')
"

# Taille DB (SQLite)
du -h backend/instance/development.db
```

---

**Toutes les commandes sont prêtes ! Copiez-collez directement. 🚀**
