# 🔄 Comportement des Rollbacks SQLAlchemy - Documentation

Ce document décrit le comportement attendu des rollbacks SQLAlchemy dans le projet ATMR, avec des exemples pratiques et des bonnes pratiques.

## 📋 Vue d'ensemble

Les rollbacks SQLAlchemy sont utilisés pour :

- **Annuler des modifications non commitées** : Restaurer l'état de la DB avant un commit
- **Gérer les erreurs** : Rollback automatique en cas d'exception
- **Isoler les tests** : Rollback automatique en fin de test via savepoints

## 🔄 Comportement Attendu

### Principe Fondamental

**Un rollback restaure l'état de la DB au dernier point de commit ou savepoint.**

### Scénarios de Rollback

#### 1. Rollback Simple (Modification Non Commitée)

```python
# État initial
booking = BookingFactory(company=company, driver_id=None)
db.session.commit()  # ✅ Commit initial

# Modification
booking.driver_id = driver.id
db.session.flush()  # ⚠️ Flush assigne l'ID mais ne commit pas

# Rollback
db.session.rollback()

# ✅ Résultat : booking.driver_id est restauré à None
```

**Comportement attendu** :

- ✅ Les modifications non commitées sont annulées
- ✅ Les valeurs en DB sont restaurées aux valeurs du dernier commit
- ✅ Les objets SQLAlchemy sont expirés (nécessitent un rechargement)

#### 2. Rollback Après Commit

```python
# État initial
booking = BookingFactory(company=company, driver_id=None)
db.session.commit()  # ✅ Commit initial

# Modification et commit
booking.driver_id = driver.id
db.session.commit()  # ✅ Commit de la modification

# Rollback
db.session.rollback()

# ⚠️ Résultat : booking.driver_id reste à driver.id (déjà commité)
```

**Comportement attendu** :

- ⚠️ Les modifications déjà commitées ne sont PAS annulées
- ⚠️ Le rollback n'affecte que les modifications non commitées
- ✅ Les objets SQLAlchemy sont expirés (nécessitent un rechargement)

#### 3. Rollback avec Savepoints (Tests)

```python
# Dans un test avec fixture db (savepoint automatique)
def test_example(db, company):
    # Créer un objet dans le savepoint
    booking = BookingFactory(company=company, driver_id=None)
    db.session.commit()  # ✅ Commit dans le savepoint

    # Modification
    booking.driver_id = driver.id
    db.session.flush()

    # Rollback
    db.session.rollback()

    # ✅ Résultat : booking.driver_id est restauré à None
    # ✅ Le savepoint est automatiquement rollback en fin de test
```

**Comportement attendu** :

- ✅ Les modifications dans le savepoint sont annulées
- ✅ Les objets commités dans le savepoint restent visibles jusqu'à la fin du test
- ✅ Le savepoint est automatiquement rollback en fin de test (isolation)

#### 4. Rollback Défensif (engine.run())

```python
# engine.run() effectue un rollback défensif au début
def test_dispatch(db, company):
    # Créer un objet et committer
    booking = BookingFactory(company=company, driver_id=None)
    db.session.commit()  # ✅ Commit avant engine.run()

    # Modification non commitée
    booking.driver_id = driver.id
    db.session.flush()

    # Appeler engine.run() qui fait un rollback défensif
    result = engine.run(company_id=company.id, ...)

    # ✅ Résultat : booking.driver_id est restauré à None (rollback défensif)
    # ✅ Les objets commités avant engine.run() restent visibles
```

**Comportement attendu** :

- ✅ Le rollback défensif annule les modifications non commitées
- ✅ Les objets commités avant l'appel restent visibles
- ✅ Le rollback défensif garantit un état de session propre

## ⚠️ Points d'Attention

### 1. Expiration des Objets

Après un rollback, les objets SQLAlchemy sont **expirés** et nécessitent un rechargement :

```python
# ❌ MAUVAIS : L'objet est expiré après rollback
db.session.rollback()
assert booking.driver_id is None  # ⚠️ Peut échouer (objet expiré)

# ✅ BON : Recharger depuis la DB
db.session.rollback()
db.session.expire_all()
booking_reloaded = db.session.query(Booking).filter_by(id=booking.id).first()
assert booking_reloaded.driver_id is None  # ✅ Correct
```

### 2. Flush vs Commit

**Flush** assigne les IDs mais ne commit pas :

```python
# Flush : ID assigné mais pas commité
booking = BookingFactory(company=company)
db.session.add(booking)
db.session.flush()  # ✅ ID assigné
assert booking.id is not None  # ✅ ID disponible

# Rollback annule même après flush
db.session.rollback()
# ⚠️ booking.id peut être None si l'objet n'a jamais été commité
```

**Commit** persiste les modifications en DB :

```python
# Commit : Modifications persistées
booking = BookingFactory(company=company)
db.session.commit()  # ✅ Persisté en DB

# Rollback n'annule pas les commits
db.session.rollback()
# ✅ booking reste en DB (déjà commité)
```

### 3. Rollback Partiel

Un rollback n'annule que les modifications non commitées :

```python
# Commit initial
booking1 = BookingFactory(company=company, driver_id=None)
booking2 = BookingFactory(company=company, driver_id=None)
db.session.commit()

# Modifier booking1 et committer
booking1.driver_id = driver.id
db.session.commit()  # ✅ booking1 modifié et commité

# Modifier booking2 mais ne pas committer
booking2.driver_id = driver.id
db.session.flush()

# Rollback
db.session.rollback()

# ✅ booking1 reste modifié (déjà commité)
# ✅ booking2 est restauré (non commité)
```

## 🧪 Vérification des Rollbacks

### Helper de Vérification

Utiliser `verify_rollback_restores_values()` pour vérifier systématiquement les rollbacks :

```python
from tests.helpers.rollback_verification import (
    capture_original_values,
    verify_rollback_restores_values,
)

# Capturer les valeurs originales
booking = BookingFactory(company=company, driver_id=None)
db.session.commit()
original_values = capture_original_values(booking, ["driver_id", "status"])

# Modifier
booking.driver_id = driver.id
db.session.flush()

# Rollback
db.session.rollback()

# Vérifier
verify_rollback_restores_values(
    db.session,
    Booking,
    booking.id,
    original_values,
)
```

### Tests de Non-Régression

Les tests suivants vérifient le comportement des rollbacks :

- `test_rollback_restores_original_values` - Vérifie qu'un rollback restaure les valeurs
- `test_rollback_restores_single_field` - Vérifie un champ unique
- `test_rollback_restores_multiple_fields` - Vérifie plusieurs champs
- `test_rollback_restores_multiple_objects` - Vérifie plusieurs objets
- `test_rollback_restores_after_flush` - Vérifie après flush
- `test_rollback_restores_after_partial_commit` - Vérifie après commit partiel
- `test_rollback_restores_after_engine_run_rollback_defensive` - Vérifie après rollback défensif

**Voir** : `backend/tests/e2e/test_rollback_robustness.py` pour tous les tests.

## 📝 Bonnes Pratiques

### 1. Dans les Tests

✅ **À faire** :

- Utiliser `verify_rollback_restores_values()` pour vérifier les rollbacks
- Capturer les valeurs originales avec `capture_original_values()` avant modification
- Recharger les objets depuis la DB après rollback (`expire_all()` + `query()`)

❌ **À éviter** :

- Ne pas vérifier les valeurs d'objets expirés sans rechargement
- Ne pas supposer que les rollbacks annulent les commits
- Ne pas oublier `expire_all()` après rollback

### 2. Dans le Code Métier

✅ **À faire** :

- Utiliser les context managers (`db_transaction()`) pour gérer automatiquement les rollbacks
- Committer explicitement les modifications importantes
- Gérer les erreurs avec rollback automatique

❌ **À éviter** :

- Ne pas faire de rollback manuel sans vérification
- Ne pas supposer que les rollbacks annulent les commits
- Ne pas oublier de gérer les objets expirés

### 3. Rollback Défensif

✅ **À faire** :

- Committer les objets avant d'appeler `engine.run()`
- Utiliser les fixtures qui garantissent le commit
- Vérifier que les objets commités restent visibles après rollback défensif

❌ **À éviter** :

- Ne pas appeler `engine.run()` avec des objets non commités
- Ne pas supposer que le rollback défensif n'affecte pas les objets commités
- Ne pas oublier de recharger les objets après rollback défensif

## 🔗 Références

- [Gestion des Sessions SQLAlchemy](./SESSION_MANAGEMENT.md) - Guide complet de gestion des sessions
- [Guide des Fixtures et Isolation](../tests/README_FIXTURES.md) - Documentation des fixtures
- [Tests de Non-Régression](../tests/README_NON_REGRESSION.md) - Tests critiques
- [SQLAlchemy Session Management](https://docs.sqlalchemy.org/en/14/orm/session_transaction.html) - Documentation officielle

---

**Note** : Ce document doit être mis à jour si de nouveaux comportements de rollback sont ajoutés.
