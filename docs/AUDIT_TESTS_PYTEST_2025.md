# Audit Complet des Tests Pytest - ATMR Backend

**Date** : 2025-01-28  
**Outils** : Pytest 9.0.1  
**Total de tests** : 3003 collectés  
**Résultats** : 5 FAILED, 5 ERROR, 41 PASSED, 1 SKIPPED

---

## 📊 Vue d'ensemble

| Statut     | Nombre | Pourcentage | Priorité    |
| ---------- | ------ | ----------- | ----------- |
| ✅ PASSED  | 41     | 80.4%       | -           |
| ❌ FAILED  | 5      | 9.8%        | 🔴 CRITIQUE |
| ⚠️ ERROR   | 5      | 9.8%        | 🔴 CRITIQUE |
| ⏭️ SKIPPED | 1      | 2.0%        | 🟢 FAIBLE   |

---

## 🔴 PROBLÈMES CRITIQUES

### 1. Fixture `company` manquante dans `test_rollback_robustness.py`

**Impact** : 🔴 CRITIQUE - 5 tests en ERROR  
**Fichier** : `backend/tests/e2e/test_rollback_robustness.py`  
**Lignes** : 24, 55, 88, 128, 159

#### Problème

Les tests utilisent la fixture `company` qui n'est pas définie dans ce fichier :

```python
def test_rollback_restores_single_field(self, db, company):  # ❌ fixture 'company' not found
```

#### Fixtures disponibles

D'après l'erreur pytest, les fixtures disponibles incluent :

- ✅ `sample_company` (définie dans `conftest.py`)
- ❌ `company` (définie dans `test_dispatch_e2e.py`, non accessible)

#### Cause racine

La fixture `company` est définie localement dans `test_dispatch_e2e.py` (ligne 24) et n'est pas accessible depuis `test_rollback_robustness.py`. Les fixtures pytest sont scopées au fichier où elles sont définies, sauf si elles sont dans `conftest.py`.

#### Solution

**Option 1 (Recommandée)** : Déplacer la fixture `company` dans `conftest.py`

```python
# backend/tests/conftest.py
@pytest.fixture
def company(db):
    """Créer une entreprise pour les tests."""
    from models import Company
    from tests.factories import CompanyFactory

    company = CompanyFactory()
    db.session.add(company)
    db.session.flush()
    db.session.commit()
    db.session.expire(company)
    company = db.session.query(Company).get(company.id)
    assert company is not None, "Company must be persisted before use"
    return company
```

**Option 2** : Utiliser `sample_company` existante

```python
# backend/tests/e2e/test_rollback_robustness.py
def test_rollback_restores_single_field(self, db, sample_company):
    company = sample_company  # Alias pour compatibilité
    # ... reste du test
```

**Option 3** : Créer une fixture locale dans `test_rollback_robustness.py`

```python
# backend/tests/e2e/test_rollback_robustness.py
import pytest
from tests.factories import CompanyFactory

@pytest.fixture
def company(db):
    """Créer une entreprise pour les tests."""
    from models import Company

    company = CompanyFactory()
    db.session.add(company)
    db.session.flush()
    db.session.commit()
    db.session.expire(company)
    company = db.session.query(Company).get(company.id)
    return company
```

**Estimation** : 15 minutes

---

### 2. Problèmes d'isolation des fixtures : Bookings associés aux mauvaises companies

**Impact** : 🔴 CRITIQUE - 2 tests FAILED  
**Tests affectés** :

- `test_rollback_transactionnel_complet` : Booking 28 appartient à company 30 au lieu de 28
- `test_apply_assignments_finds_bookings` : Booking 63 appartient à company 76 au lieu de 74

#### Problème

Les bookings créés par les fixtures sont associés aux mauvaises companies, suggérant un problème d'isolation entre les tests ou un problème dans l'ordre d'exécution des fixtures.

#### Exemple d'erreur

```
AssertionError: Booking 28 must belong to company 28, got 30
assert 30 == 28
 +  where 30 = <Booking 28>.company_id
 +  and   28 = <Company Lemonnier | ID: 28 | Approved: True>.id
```

#### Cause racine possible

1. **Isolation insuffisante** : Les objets créés dans un test sont visibles dans un autre test
2. **Ordre d'exécution des fixtures** : La fixture `bookings` pourrait être créée avant que `company` soit correctement persistée
3. **Problème de savepoint** : Les savepoints ne sont pas correctement isolés

#### Solution

**Vérifier l'isolation des fixtures** :

```python
# backend/tests/e2e/test_dispatch_e2e.py
@pytest.fixture
def bookings(db, company, drivers):
    """Créer plusieurs bookings pour les tests."""
    # ✅ S'assurer que company est bien persistée
    db.session.refresh(company)  # Recharger depuis DB
    assert company.id is not None, "Company must be persisted"

    bookings = []
    for i in range(5):
        booking = BookingFactory(company=company, driver_id=None)
        db.session.add(booking)
        db.session.flush()  # Force l'assignation de l'ID
        # ✅ Vérifier que booking.company_id == company.id
        assert booking.company_id == company.id, (
            f"Booking {booking.id} must belong to company {company.id}, "
            f"got {booking.company_id}"
        )
        bookings.append(booking)

    db.session.commit()

    # ✅ Recharger depuis DB pour garantir persistance
    for booking in bookings:
        db.session.expire(booking)
        booking_from_db = db.session.query(Booking).get(booking.id)
        assert booking_from_db is not None
        assert booking_from_db.company_id == company.id
        bookings[bookings.index(booking)] = booking_from_db

    return bookings
```

**Estimation** : 30 minutes

---

### 3. Problèmes de données manquantes : "no_bookings", "no_drivers", "no_data"

**Impact** : 🔴 CRITIQUE - 3 tests FAILED  
**Tests affectés** :

- `test_dispatch_async_complet` : "no_bookings"
- `test_validation_temporelle_stricte_rollback` : "no_drivers"
- `test_dispatch_run_id_correlation` : "no_bookings"

#### Problème

Les tests échouent car les données nécessaires (bookings, drivers) ne sont pas disponibles pour le dispatch.

#### Exemple d'erreur

```
WARNING  services.unified_dispatch.data:data.py:1358 [Dispatch] No dispatch possible for company 4: no_bookings
assert 0 > 0
 +  where 0 = len([])
```

#### Cause racine possible

1. **Fixtures non persistées** : Les bookings/drivers ne sont pas correctement commités avant l'appel à `engine.run()`
2. **Filtrage trop strict** : Les bookings sont filtrés par `data.py` (retours non confirmés, etc.)
3. **Problème de timing** : Les objets ne sont pas visibles dans le savepoint utilisé par `engine.run()`

#### Solution

**S'assurer que les fixtures sont persistées** :

```python
# backend/tests/e2e/test_dispatch_e2e.py
def test_dispatch_async_complet(self, company, drivers, bookings, db):
    """Test : Dispatch asynchrone complet."""
    # ✅ FIX: S'assurer que tout est commité avant engine.run()
    db.session.commit()

    # ✅ FIX: Vérifier que les données existent en DB
    bookings_from_db = db.session.query(Booking).filter_by(company_id=company.id).all()
    assert len(bookings_from_db) > 0, "Bookings must exist in DB"

    drivers_from_db = db.session.query(Driver).filter_by(company_id=company.id).all()
    assert len(drivers_from_db) > 0, "Drivers must exist in DB"

    # ✅ FIX: Vérifier que les bookings ne sont pas filtrés
    # (pas de retour non confirmé, etc.)
    for booking in bookings:
        assert booking.return_time is None or booking.return_time_confirmed is True, (
            f"Booking {booking.id} should not be filtered"
        )

    # Appeler engine.run()
    result = engine.run(company_id=company.id, for_date=date.today().isoformat())
    # ... reste du test
```

**Estimation** : 45 minutes

---

### 4. Raison de dispatch incorrecte : "no_data" au lieu de "run_failed"

**Impact** : 🟡 MOYENNE - 1 test FAILED  
**Test affecté** : `test_validation_temporelle_stricte_rollback`

#### Problème

Le test s'attend à ce que le dispatch échoue avec `reason` dans `["run_failed", "validation_failed", "conflict"]`, mais obtient `"no_data"`.

#### Exemple d'erreur

```
AssertionError: Le dispatch devrait avoir échoué, mais reason=no_data
assert 'no_data' in ['run_failed', 'validation_failed', 'conflict']
```

#### Cause racine

Le dispatch échoue avec `"no_data"` (pas de drivers) au lieu d'échouer avec une raison de validation/conflict. Cela suggère que le test ne configure pas correctement les données pour déclencher l'erreur attendue.

#### Solution

**Configurer correctement les données pour déclencher l'erreur attendue** :

```python
# backend/tests/e2e/test_dispatch_e2e.py
def test_validation_temporelle_stricte_rollback(self, company, drivers, bookings, db):
    """Test : Validation temporelle stricte avec rollback."""
    # ✅ FIX: S'assurer que les drivers existent
    assert len(drivers) > 0, "Drivers must exist for this test"

    # ✅ FIX: Configurer les bookings pour déclencher une erreur de validation
    # Par exemple : créer des bookings avec des conflits temporels
    for booking in bookings:
        booking.pickup_time = datetime.now() + timedelta(hours=1)
        booking.return_time = datetime.now() + timedelta(hours=2)
        # Créer un conflit temporel
        booking.return_time_confirmed = False  # Retour non confirmé

    db.session.commit()

    # Appeler engine.run() qui devrait échouer avec validation_failed
    result = engine.run(company_id=company.id, for_date=date.today().isoformat())

    # Vérifier que le résultat indique un problème
    if result.get("meta", {}).get("reason"):
        assert result["meta"]["reason"] in ["run_failed", "validation_failed", "conflict"], (
            f"Le dispatch devrait avoir échoué, mais reason={result['meta'].get('reason')}"
        )
```

**Estimation** : 30 minutes

---

## 📋 Plan d'Action Recommandé

### Phase 1 : Corrections Critiques (Immédiat - 1h)

1. **Corriger la fixture `company` manquante** (15 min)

   - Déplacer `company` dans `conftest.py` ou utiliser `sample_company`
   - Tester que les 5 tests ERROR passent

2. **Corriger l'isolation des fixtures** (30 min)

   - Ajouter des vérifications dans `bookings` fixture
   - S'assurer que `company_id` est correctement assigné
   - Tester que les 2 tests FAILED (isolation) passent

3. **Corriger les problèmes de données manquantes** (15 min)
   - S'assurer que les fixtures sont commitées avant `engine.run()`
   - Ajouter des vérifications que les données existent en DB
   - Tester que les 3 tests FAILED (données) passent

### Phase 2 : Corrections Moyennes (Optionnel - 30 min)

1. **Corriger la raison de dispatch** (30 min)
   - Configurer correctement les données pour déclencher l'erreur attendue
   - Tester que `test_validation_temporelle_stricte_rollback` passe

### Phase 3 : Validation

1. **Exécuter tous les tests** : `pytest backend/tests/e2e/ -v`
2. **Vérifier que tous les tests passent** : 0 FAILED, 0 ERROR
3. **Documenter les corrections** dans ce fichier

---

## 📊 Résultats Attendus

| Avant                    | Après                   | Amélioration |
| ------------------------ | ----------------------- | ------------ |
| 5 FAILED                 | 0 FAILED                | ✅ 100%      |
| 5 ERROR                  | 0 ERROR                 | ✅ 100%      |
| 41 PASSED                | 51 PASSED               | ✅ +24%      |
| Taux de réussite : 80.4% | Taux de réussite : 100% | ✅ +19.6%    |

---

## 🔧 Fichiers à Modifier

1. **`backend/tests/conftest.py`** : Ajouter fixture `company` globale
2. **`backend/tests/e2e/test_dispatch_e2e.py`** : Corriger fixtures `bookings` et tests
3. **`backend/tests/e2e/test_rollback_robustness.py`** : Utiliser fixture `company` depuis `conftest.py`

---

## ✅ Conclusion

**Problèmes principaux** :

1. Fixture `company` manquante (5 ERROR)
2. Isolation insuffisante des fixtures (2 FAILED)
3. Données manquantes pour dispatch (3 FAILED)

**Solution** : Déplacer la fixture `company` dans `conftest.py`, améliorer l'isolation des fixtures, et s'assurer que les données sont correctement persistées avant `engine.run()`.

**Résultat final attendu** : 100% de tests passants (51/51 au lieu de 41/51).
