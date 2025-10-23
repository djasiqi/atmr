# 🚀 SEMAINE 2 - GUIDE DÉTAILLÉ : Optimisations Base de Données

**Période** : Jour 1 à Jour 5  
**Objectif** : Optimiser les performances de la base de données  
**Livrable** : -50% temps queries, +Performance SQL massive

---

## 📋 VUE D'ENSEMBLE SEMAINE 2

| Jour       | Tâche Principale | Effort | Impact              |
| ---------- | ---------------- | ------ | ------------------- |
| **Jour 1** | Profiling DB     | 6h     | Identifier goulots  |
| **Jour 2** | Index DB         | 6h     | -50% temps queries  |
| **Jour 3** | Bulk inserts     | 6h     | -90% temps écriture |
| **Jour 4** | Queries N+1      | 6h     | -67% nombre queries |
| **Jour 5** | Validation       | 6h     | Confirmer gains     |

**Total effort** : 30 heures (1 semaine pour 1 développeur)

---

## 📅 JOUR 1 : Profiling Base de Données

### Objectif

Identifier les requêtes SQL lentes et les goulots d'étranglement de performance.

### Prérequis

- Base de données avec données de test (>100 bookings, >20 drivers)
- Flask application fonctionnelle

### Étapes Détaillées

#### Étape 1.1 : Installer Outils de Profiling (1h)

```bash
cd backend

# Installer packages
.\venv\Scripts\python.exe -m pip install flask-sqlalchemy
.\venv\Scripts\python.exe -m pip install sqlalchemy-utils
.\venv\Scripts\python.exe -m pip install nplusone

# Ajouter à requirements.txt
echo "nplusone==1.0.0" >> requirements.txt
```

#### Étape 1.2 : Activer SQL Logging (30min)

Modifier `backend/config.py` :

```python
# Ajouter dans la classe Config
class Config:
    # ... existing config ...

    # SQL Logging (pour dev/profiling uniquement)
    SQLALCHEMY_ECHO = os.getenv('SQL_ECHO', 'False').lower() == 'true'
    SQLALCHEMY_RECORD_QUERIES = True

    # Slow query threshold
    DATABASE_QUERY_TIMEOUT = 0.1  # 100ms
```

Créer `.env.profiling` :

```bash
SQL_ECHO=true
FLASK_DEBUG=true
```

#### Étape 1.3 : Créer Script de Profiling (2h)

Créer `backend/scripts/profiling/profile_dispatch.py` :

```python
"""
Script de profiling pour mesurer les performances du dispatch.
"""
import time
from datetime import date
from flask import Flask
from sqlalchemy import event
from sqlalchemy.engine import Engine

# Liste des queries exécutées
queries_log = []

@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total_time = time.time() - context._query_start_time

    # Logger queries > 50ms
    if total_time > 0.050:
        queries_log.append({
            'query': statement[:200],  # Limiter taille
            'time': total_time,
            'params': str(parameters)[:100]
        })

def profile_dispatch(company_id=1, for_date=None):
    """Profile un dispatch complet."""
    from app import create_app
    from services.unified_dispatch.engine import DispatchEngine

    app = create_app()

    with app.app_context():
        global queries_log
        queries_log = []

        if for_date is None:
            for_date = date.today().isoformat()

        # Mesurer temps total
        start_time = time.time()

        # Exécuter dispatch
        engine = DispatchEngine()
        result = engine.run(company_id=company_id, for_date=for_date)

        end_time = time.time()
        total_time = end_time - start_time

        # Analyser résultats
        print("\n" + "="*60)
        print("📊 PROFILING RESULTS - DISPATCH")
        print("="*60)
        print(f"\n⏱️  Temps total : {total_time:.2f}s")
        print(f"📦 Assignments créés : {len(result.get('assignments', []))}")
        print(f"🔍 Queries lentes (>50ms) : {len(queries_log)}")

        if queries_log:
            print("\n🐌 TOP 10 QUERIES LES PLUS LENTES:")
            sorted_queries = sorted(queries_log, key=lambda x: x['time'], reverse=True)

            for i, q in enumerate(sorted_queries[:10], 1):
                print(f"\n{i}. Temps: {q['time']*1000:.1f}ms")
                print(f"   Query: {q['query']}")
                if q['params']:
                    print(f"   Params: {q['params']}")

        # Sauvegarder rapport
        with open('scripts/profiling/profiling_results.txt', 'w') as f:
            f.write(f"Temps total: {total_time:.2f}s\n")
            f.write(f"Queries lentes: {len(queries_log)}\n\n")

            for i, q in enumerate(sorted_queries, 1):
                f.write(f"\n{i}. {q['time']*1000:.1f}ms - {q['query']}\n")

        print(f"\n💾 Rapport sauvegardé dans scripts/profiling/profiling_results.txt")

        return {
            'total_time': total_time,
            'slow_queries': len(queries_log),
            'queries': sorted_queries
        }

if __name__ == "__main__":
    profile_dispatch(company_id=1)
```

**Utilisation** :

```bash
cd backend
.\venv\Scripts\python.exe scripts/profiling/profile_dispatch.py
```

#### Étape 1.4 : Analyser Résultats (1h)

Créer `backend/scripts/profiling/PROFILING_RESULTS.md` :

```markdown
# Profiling Results - Baseline

**Date** : [DATE]
**Database** : SQLite/PostgreSQL

## Métriques Baseline

- **Temps total dispatch** : **\_** secondes
- **Nombre queries** : **\_**
- **Queries lentes (>100ms)** : **\_**

## Top 10 Queries Lentes

1. Query : ...
   Temps : \_\_\_ ms
   Fichier : ...
   Action : Ajouter index sur ...

2. Query : ...
   Temps : \_\_\_ ms
   ...
```

#### Étape 1.5 : Créer Plan d'Index (1h)

D'après les queries lentes, créer liste d'index à ajouter :

```markdown
# Index à Créer

## Priorité 1 (Critique - Jour 2)

1. `idx_assignment_booking_created`

   - Table: assignment
   - Colonnes: (booking_id, created_at DESC)
   - Raison: Query lente sur récupération assignments

2. `idx_booking_status_scheduled_company`

   - Table: booking
   - Colonnes: (status, scheduled_time, company_id)
   - Raison: Filtre bookings pour dispatch

3. `idx_driver_available_company`
   - Table: driver
   - Colonnes: (company_id) WHERE is_available=true AND is_active=true
   - Raison: Récupération drivers disponibles

...
```

**Livrable Jour 1** :

- ✅ Rapport profiling complet
- ✅ Top 10 queries lentes identifiées
- ✅ Plan d'index créé

---

## 📅 JOUR 2 : Créer Index Base de Données

### Objectif

Créer les index manquants pour accélérer les queries identifiées.

### Étapes Détaillées

#### Étape 2.1 : Créer Migration Alembic (2h)

```bash
cd backend

# Créer nouvelle migration
.\venv\Scripts\python.exe -m flask db revision -m "add_performance_indexes"

# Fichier créé : migrations/versions/xxx_add_performance_indexes.py
```

Éditer le fichier de migration :

```python
"""add_performance_indexes

Revision ID: xxx
Revises: yyy
Create Date: 2025-10-20

Ajoute les index de performance identifiés par profiling.
"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    # Index 1: Assignment - booking_id + created_at
    op.create_index(
        'idx_assignment_booking_created',
        'assignment',
        ['booking_id', sa.text('created_at DESC')],
        unique=False
    )

    # Index 2: Booking - status + scheduled_time + company_id
    op.create_index(
        'idx_booking_status_scheduled_company',
        'booking',
        ['status', 'scheduled_time', 'company_id'],
        unique=False
    )

    # Index 3: Driver - company_id (avec filtre pour disponibles)
    # Note: Filtered index pas supporté par SQLite, créer index complet
    op.create_index(
        'idx_driver_company_available',
        'driver',
        ['company_id', 'is_available', 'is_active'],
        unique=False
    )

    # Index 4: Booking - company_id + scheduled_time
    op.create_index(
        'idx_booking_company_scheduled',
        'booking',
        ['company_id', 'scheduled_time'],
        unique=False
    )

    # Index 5: Assignment - dispatch_run_id + status
    op.create_index(
        'idx_assignment_run_status',
        'assignment',
        ['dispatch_run_id', 'status'],
        unique=False
    )

def downgrade():
    # Supprimer dans l'ordre inverse
    op.drop_index('idx_assignment_run_status', table_name='assignment')
    op.drop_index('idx_booking_company_scheduled', table_name='booking')
    op.drop_index('idx_driver_company_available', table_name='driver')
    op.drop_index('idx_booking_status_scheduled_company', table_name='booking')
    op.drop_index('idx_assignment_booking_created', table_name='assignment')
```

#### Étape 2.2 : Tester Migration (1h)

```bash
# Appliquer migration
.\venv\Scripts\python.exe -m flask db upgrade

# Vérifier index créés (SQLite)
.\venv\Scripts\python.exe -c "from ext import db; print(db.engine.execute('PRAGMA index_list(assignment)').fetchall())"

# Tester downgrade
.\venv\Scripts\python.exe -m flask db downgrade

# Re-upgrade
.\venv\Scripts\python.exe -m flask db upgrade
```

#### Étape 2.3 : Benchmark Avant/Après (2h)

Créer `backend/scripts/profiling/benchmark_indexes.py` :

```python
"""Benchmark performance avant/après index."""
import time
from app import create_app
from services.unified_dispatch.engine import DispatchEngine

def benchmark(iterations=5):
    app = create_app()

    with app.app_context():
        times = []

        for i in range(iterations):
            start = time.time()
            engine = DispatchEngine()
            result = engine.run(company_id=1, for_date="2025-10-20")
            end = time.time()

            elapsed = end - start
            times.append(elapsed)
            print(f"Run {i+1}: {elapsed:.2f}s")

        avg = sum(times) / len(times)
        print(f"\n📊 Moyenne : {avg:.2f}s")

        return avg

if __name__ == "__main__":
    print("Benchmark avec index...")
    benchmark()
```

Lancer avant et après :

```bash
# AVANT index
.\venv\Scripts\python.exe -m flask db downgrade  # Retirer index
.\venv\Scripts\python.exe scripts/profiling/benchmark_indexes.py
# Noter le temps

# APRÈS index
.\venv\Scripts\python.exe -m flask db upgrade  # Ajouter index
.\venv\Scripts\python.exe scripts/profiling/benchmark_indexes.py
# Noter le temps

# Calculer gain
```

**Livrable Jour 2** :

- ✅ Migration Alembic créée
- ✅ 5-10 index DB créés
- ✅ Benchmark avant/après documenté
- ✅ Gain de performance mesuré

---

## 📅 JOUR 3 : Bulk Inserts

### Objectif

Remplacer les boucles avec commits multiples par des bulk inserts.

### Problème Actuel

Dans `apply.py`, le code fait :

```python
# ❌ LENT (N commits)
for assignment in assignments:
    db_assignment = Assignment(...)
    db.session.add(db_assignment)
    db.session.commit()  # Commit à chaque itération
```

**Problème** : Si 50 assignments → 50 commits → très lent !

### Solution : Bulk Insert

```python
# ✅ RAPIDE (1 commit)
assignment_dicts = []
for assignment in assignments:
    assignment_dicts.append({
        'booking_id': assignment.booking_id,
        'driver_id': assignment.driver_id,
        ...
    })

db.session.bulk_insert_mappings(Assignment, assignment_dicts)
db.session.commit()  # 1 seul commit
```

### Étapes Détaillées

#### Étape 3.1 : Analyser apply.py (1h)

```bash
cd backend/services/unified_dispatch

# Trouver tous les commits dans apply.py
grep -n "commit()" apply.py

# Lire la fonction
code apply.py
# Chercher _apply_and_emit
```

#### Étape 3.2 : Backup (5min)

```bash
cp apply.py apply.py.backup
```

#### Étape 3.3 : Refactoriser (3h)

Ouvrir `apply.py` et trouver la section qui crée les assignments.

**AVANT** :

```python
for assignment_data in final_assignments:
    # Créer objet Assignment
    assignment = Assignment(
        booking_id=assignment_data.booking_id,
        driver_id=assignment_data.driver_id,
        dispatch_run_id=dispatch_run_id,
        status='pending',
        confirmed=False
    )
    db.session.add(assignment)
    db.session.commit()  # ❌ Commit multiple

    # Update booking status
    booking = Booking.query.get(assignment_data.booking_id)
    booking.status = BookingStatus.ASSIGNED
    db.session.commit()  # ❌ Encore un commit
```

**APRÈS** :

```python
# Phase 1 : Bulk insert assignments
assignment_mappings = []
booking_ids_to_update = []

for assignment_data in final_assignments:
    assignment_mappings.append({
        'booking_id': assignment_data.booking_id,
        'driver_id': assignment_data.driver_id,
        'dispatch_run_id': dispatch_run_id,
        'status': 'pending',
        'confirmed': False,
        'created_at': datetime.now(UTC)
    })
    booking_ids_to_update.append(assignment_data.booking_id)

# Bulk insert (1 seule transaction)
db.session.bulk_insert_mappings(Assignment, assignment_mappings)

# Phase 2 : Bulk update bookings
if booking_ids_to_update:
    Booking.query.filter(
        Booking.id.in_(booking_ids_to_update)
    ).update(
        {Booking.status: BookingStatus.ASSIGNED},
        synchronize_session=False
    )

# 1 seul commit pour tout !
db.session.commit()  # ✅ Commit unique
```

#### Étape 3.4 : Tests (1h30)

Créer `backend/tests/test_apply_bulk.py` :

```python
"""Tests pour bulk inserts dans apply.py"""
import pytest
from datetime import datetime, UTC
from models import Assignment, Booking, BookingStatus
from services.unified_dispatch.apply import apply_assignments

def test_bulk_insert_creates_all_assignments(app, db_session):
    """Bulk insert crée tous les assignments."""
    # Créer données test
    assignments_data = [
        {'booking_id': 1, 'driver_id': 10},
        {'booking_id': 2, 'driver_id': 11},
        {'booking_id': 3, 'driver_id': 12},
    ]

    # Appliquer
    apply_assignments(assignments_data, dispatch_run_id=100)

    # Vérifier
    assignments = Assignment.query.filter_by(dispatch_run_id=100).all()
    assert len(assignments) == 3

def test_bulk_insert_updates_booking_status(app, db_session):
    """Bulk insert met à jour status des bookings."""
    # ... test ici
```

Lancer tests :

```bash
.\venv\Scripts\python.exe -m pytest tests/test_apply_bulk.py -v
```

**Livrable Jour 3** :

- ✅ apply.py refactorisé (bulk inserts)
- ✅ Tests passent
- ✅ -90% temps écriture DB

---

## 📅 JOUR 4 : Éliminer Queries N+1

### Objectif

Éliminer les queries N+1 (requêtes inutiles dans des boucles).

### Problème N+1

```python
# ❌ N+1 Problem
bookings = Booking.query.filter_by(company_id=1).all()  # 1 query

for booking in bookings:
    client_name = booking.client.name  # N queries supplémentaires !
```

Si 50 bookings → 1 + 50 = **51 queries** au lieu de 1 !

### Solution : Eager Loading

```python
# ✅ Solution
bookings = Booking.query.filter_by(company_id=1)\
    .options(joinedload(Booking.client))\
    .all()  # 1 seule query avec JOIN

for booking in bookings:
    client_name = booking.client.name  # Pas de query !
```

### Étapes Détaillées

#### Étape 4.1 : Détecter N+1 (2h)

Installer détecteur :

```bash
.\venv\Scripts\python.exe -m pip install nplusone
```

Activer dans `app.py` :

```python
from nplusone.ext.flask_sqlalchemy import NPlusOne

app = create_app()
NPlusOne(app)
```

Lancer dispatch et voir warnings.

#### Étape 4.2 : Corriger Routes (3h)

**Fichier : `routes/bookings.py`**

AVANT :

```python
@bookings_bp.route('/', methods=['GET'])
def get_bookings():
    bookings = Booking.query.filter_by(company_id=company_id).all()
    return jsonify([b.serialize() for b in bookings])
```

APRÈS :

```python
from sqlalchemy.orm import joinedload

@bookings_bp.route('/', methods=['GET'])
def get_bookings():
    bookings = Booking.query\
        .filter_by(company_id=company_id)\
        .options(
            joinedload(Booking.client),
            joinedload(Booking.company)
        )\
        .all()
    return jsonify([b.serialize() for b in bookings])
```

Faire de même pour :

- `routes/dispatch_routes.py`
- `routes/drivers.py`
- `services/unified_dispatch/data.py`

#### Étape 4.3 : Vérifier (1h)

```bash
# Compter queries
.\venv\Scripts\python.exe scripts/profiling/profile_dispatch.py

# Avant : ~150 queries
# Après : ~50 queries (-67%)
```

**Livrable Jour 4** :

- ✅ Queries N+1 éliminées
- ✅ Routes optimisées avec joinedload
- ✅ -67% nombre queries

---

## 📅 JOUR 5 : Tests Performance et Validation

### Objectif

Mesurer tous les gains et valider que tout fonctionne.

### Étapes Détaillées

#### Étape 5.1 : Benchmark Final (2h)

Créer `backend/scripts/profiling/final_benchmark.py` :

```python
"""Benchmark final comparant avant/après Semaine 2."""
import time
import statistics

def benchmark_comprehensive():
    results = {
        'dispatch_time': [],
        'query_count': [],
        'slow_queries': []
    }

    for i in range(10):
        # Mesurer
        start = time.time()
        result = run_dispatch()
        end = time.time()

        results['dispatch_time'].append(end - start)
        # ... collecter autres métriques

    # Statistiques
    print(f"Temps moyen : {statistics.mean(results['dispatch_time']):.2f}s")
    print(f"Écart-type  : {statistics.stdev(results['dispatch_time']):.2f}s")
    print(f"Min         : {min(results['dispatch_time']):.2f}s")
    print(f"Max         : {max(results['dispatch_time']):.2f}s")

if __name__ == "__main__":
    benchmark_comprehensive()
```

#### Étape 5.2 : Créer Graphiques (2h)

```python
import matplotlib.pyplot as plt

# Graphique avant/après
categories = ['Dispatch\nTime', 'Query\nCount', 'Slow\nQueries']
avant = [45, 150, 15]
apres = [20, 50, 3]

x = range(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar([i - width/2 for i in x], avant, width, label='Avant', color='red')
bars2 = ax.bar([i + width/2 for i in x], apres, width, label='Après', color='green')

ax.set_ylabel('Valeur')
ax.set_title('Performance DB - Avant/Après Semaine 2')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.savefig('scripts/profiling/performance_comparison.png')
print("Graphique sauvegardé")
```

#### Étape 5.3 : Tests de Charge (1h)

```python
"""Tests de charge avec beaucoup de données."""
def test_dispatch_with_100_bookings():
    # Créer 100 bookings
    # Lancer dispatch
    # Mesurer temps
    # Assert < 30s
    pass

def test_dispatch_with_1000_bookings():
    # Test de stress
    pass
```

#### Étape 5.4 : Rapport Final (1h)

Créer `backend/scripts/profiling/PERFORMANCE_REPORT.md` :

```markdown
# Performance Report - Semaine 2

## Résultats Baseline (Avant)

- Temps dispatch : 45s
- Queries : 150
- Queries lentes : 15

## Résultats Après Optimisations

- Temps dispatch : 20s (-56%)
- Queries : 50 (-67%)
- Queries lentes : 3 (-80%)

## Détail des Optimisations

### Index DB (Jour 2)

- 5 index créés
- Gain : -30% temps queries

### Bulk Inserts (Jour 3)

- apply.py refactorisé
- Gain : -90% temps écriture

### Queries N+1 (Jour 4)

- Routes optimisées
- Gain : -67% nombre queries

## Graphiques

![Performance Comparison](performance_comparison.png)

## Conclusion

✅ Tous les objectifs atteints
✅ Performance améliorée de 56%
✅ Application ultra-rapide
```

**Livrable Jour 5** :

- ✅ Rapport performance complet
- ✅ Graphiques comparatifs
- ✅ Tous les tests passent
- ✅ Documentation complète

---

## ✅ VALIDATION FINALE

### Checklist Complète

- [ ] Profiling effectué
- [ ] 5-10 index créés
- [ ] Migration testée (up + down)
- [ ] Bulk inserts implémentés
- [ ] Queries N+1 éliminées
- [ ] Benchmarks documentés
- [ ] Tous les tests passent
- [ ] -50%+ temps dispatch
- [ ] Rapport final complété

---

**FIN DU GUIDE SEMAINE 2**

**Total pages** : ~40 pages  
**Temps lecture** : ~2-3 heures  
**Temps implémentation** : 30 heures
