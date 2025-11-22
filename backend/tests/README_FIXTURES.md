# 📚 Guide des Fixtures et Isolation des Tests

Ce document décrit les bonnes pratiques pour créer et utiliser des fixtures dans les tests ATMR, avec un focus sur l'isolation et la gestion des transactions.

## 🔄 Isolation via Savepoints

Chaque test utilise un **savepoint** (nested transaction) via la fixture `db`. Cela garantit :

- ✅ Isolation complète entre les tests
- ✅ Rollback automatique en fin de test
- ✅ Pas de pollution de données entre tests

```python
@pytest.fixture
def db(app):
    """Crée une DB propre pour chaque test en utilisant des savepoints."""
    with app.app_context():
        _db.session.begin_nested()  # Créer un savepoint
        yield _db
        _db.session.rollback()  # Rollback automatique
        _db.session.expire_all()
        _db.session.remove()
```

## 🏭 Fixtures Persistées

### Problème : Rollback défensif de `engine.run()`

La fonction `engine.run()` effectue un rollback défensif au début, ce qui peut expirer les objets SQLAlchemy non commités. **TOUJOURS commit les objets avant d'appeler `engine.run()`**.

### Solution : Helper `persisted_fixture()`

Utilisez le helper `persisted_fixture()` pour créer des fixtures qui garantissent la persistance :

```python
from tests.conftest import persisted_fixture
from tests.factories import CompanyFactory
from models import Company

@pytest.fixture
def company(db):
    """Créer une entreprise persistée pour les tests."""
    return persisted_fixture(db, CompanyFactory(), Company)
```

**Avantages** :

- ✅ Commit automatique
- ✅ Rechargement depuis la DB pour garantir la persistance
- ✅ Vérification que l'objet existe
- ✅ Code réutilisable et générique

### Exemple complet

```python
@pytest.fixture
def company(db):
    """Créer une entreprise pour les tests."""
    return persisted_fixture(db, CompanyFactory(), Company)

@pytest.fixture
def drivers(db, company):
    """Créer plusieurs chauffeurs pour les tests."""
    drivers_list = []
    for _ in range(3):
        driver = persisted_fixture(
            db,
            DriverFactory(company=company, is_active=True),
            Driver
        )
        drivers_list.append(driver)
    return drivers_list
```

## 🔧 Helpers Disponibles

### `persisted_fixture(db_session, factory_instance, model_class, **kwargs)`

Crée un objet via une factory, le commit dans la DB, et le recharge pour garantir la persistance.

**Paramètres** :

- `db_session` : **Instance Flask-SQLAlchemy** (généralement la fixture `db` ou `db_session`)
  - ⚠️ **IMPORTANT** : `db_session` est l'instance Flask-SQLAlchemy (`_db`), pas la session SQLAlchemy
  - La fonction utilise `db_session.session.add()`, `db_session.session.commit()`, etc. en interne
  - Ne pas passer `db.session` directement, passer `db` (l'instance Flask-SQLAlchemy)
- `factory_instance` : Instance de factory (ex: `CompanyFactory()`)
- `model_class` : Classe du modèle SQLAlchemy (ex: `Company`)
- `reload` : Si True, expire et recharge l'objet depuis la DB (défaut: True)
- `assert_exists` : Si True, vérifie que l'objet existe après reload (défaut: True)

**Retourne** : Instance du modèle persisté et rechargé depuis la DB

**Pattern correct** :

```python
# ✅ CORRECT : Passer l'instance Flask-SQLAlchemy
company = persisted_fixture(db, CompanyFactory(), Company)

# ❌ INCORRECT : Ne pas passer db.session directement
# company = persisted_fixture(db.session, CompanyFactory(), Company)  # ERREUR
```

**Note technique** : `persisted_fixture()` utilise `db_session.session.add()` en interne car `db_session` est l'instance Flask-SQLAlchemy qui expose la session via l'attribut `.session`.

### `ensure_committed(db_session)`

Context manager pour garantir que tous les objets sont commités avant utilisation.

```python
def test_dispatch(db, company):
    with ensure_committed(db):
        # Tous les objets sont garantis commités ici
        result = engine.run(company_id=company.id, ...)
```

### `nested_savepoint(db_session)`

Context manager pour créer un savepoint imbriqué (nested transaction).

```python
def test_nested_transaction(db):
    # Créer des objets dans le savepoint principal
    obj1 = MyEntityFactory()
    db.session.add(obj1)
    db.session.commit()

    # Créer un savepoint imbriqué
    with nested_savepoint(db):
        obj2 = MyEntityFactory()
        db.session.add(obj2)
        db.session.commit()
        # obj2 sera rollback à la fin du context manager

    # obj1 existe toujours, obj2 a été rollback
    assert obj1.id is not None
```

⚠️ **Attention** : Les savepoints imbriqués sont rollback automatiquement si le savepoint parent est rollback. Ne pas utiliser pour isoler des tests (utiliser la fixture `db` à la place).

## 🔑 Patterns SQLAlchemy Corrects

### Utilisation de Flask-SQLAlchemy dans les tests

**Pattern correct** : Utiliser `db.session` pour accéder à la session SQLAlchemy

```python
# ✅ CORRECT : db est l'instance Flask-SQLAlchemy
db.session.add(obj)
db.session.commit()
db.session.query(Model).filter_by(...).first()

# ❌ INCORRECT : Ne pas utiliser db.add() directement
# db.add(obj)  # AttributeError: add n'existe pas sur l'instance Flask-SQLAlchemy
```

**Explication** :

- `db` (ou `db_session`) est l'instance Flask-SQLAlchemy (`_db` importée depuis `ext`)
- Flask-SQLAlchemy expose la session SQLAlchemy via l'attribut `.session`
- Pour accéder aux méthodes de la session (add, commit, query, etc.), utiliser `db.session.add()`, pas `db.add()`

**Dans persisted_fixture()** :

- La fonction reçoit `db` (instance Flask-SQLAlchemy)
- Elle utilise `db_session.session.add()` en interne
- C'est pourquoi il faut passer `db` et non `db.session` à `persisted_fixture()`

### Pattern pour les fixtures

```python
@pytest.fixture
def db_session(db):
    """Alias pour db pour compatibilité avec les tests existants."""
    return db  # Retourne l'instance Flask-SQLAlchemy, pas db.session

@pytest.fixture
def my_entity(db):
    # ✅ CORRECT : Passer db (instance Flask-SQLAlchemy)
    return persisted_fixture(db, MyEntityFactory(), MyEntity)

    # ❌ INCORRECT : Ne pas passer db.session
    # return persisted_fixture(db.session, MyEntityFactory(), MyEntity)  # ERREUR
```

### Pièges à éviter

1. **Ne pas utiliser `db.add()` directement** : Utiliser `db.session.add()`
2. **Ne pas passer `db.session` à `persisted_fixture()`** : Passer `db` (l'instance)
3. **Ne pas confondre `db` et `db.session`** :
   - `db` = instance Flask-SQLAlchemy
   - `db.session` = session SQLAlchemy

## 📝 Bonnes Pratiques

### 1. Toujours commit avant `engine.run()`

```python
# ✅ BON
@pytest.fixture
def company(db):
    return persisted_fixture(db, CompanyFactory(), Company)

def test_dispatch(company):
    result = engine.run(company_id=company.id)  # ✅ Company est commitée

# ❌ MAUVAIS
@pytest.fixture
def company(db):
    company = CompanyFactory()
    db.session.add(company)
    db.session.flush()  # ❌ Pas de commit
    return company

def test_dispatch(company):
    result = engine.run(company_id=company.id)  # ❌ Company peut être expirée
```

### 2. Recharger après rollback

```python
# ✅ BON
def test_rollback(db, company):
    booking = BookingFactory(company=company)
    db.session.commit()

    # Modifier
    booking.driver_id = driver.id
    db.session.flush()

    # Rollback
    db.session.rollback()
    db.session.expire_all()

    # Recharger depuis la DB
    booking_reloaded = db.session.query(Booking).filter_by(id=booking.id).first()
    assert booking_reloaded.driver_id is None  # ✅ Valeur restaurée

# ❌ MAUVAIS
def test_rollback(db, company):
    booking = BookingFactory(company=company)
    db.session.commit()

    booking.driver_id = driver.id
    db.session.rollback()

    # ❌ Ne pas réutiliser l'objet expiré sans recharger
    assert booking.driver_id is None  # ❌ Peut échouer
```

### 3. Utiliser `query.filter_by().first()` après rollback

```python
# ✅ BON
booking_reloaded = db.session.query(Booking).filter_by(id=booking.id).first()

# ⚠️ ATTENTION
booking_reloaded = db.session.query(Booking).get(booking.id)  # Peut retourner None si expiré
```

### 4. Documenter les dépendances entre fixtures

```python
@pytest.fixture
def drivers(db, company):
    """Créer plusieurs chauffeurs pour les tests.

    ⚠️ DÉPENDANCE :
    - Dépend de la fixture `company` (ordre d'exécution garanti par pytest)
    - La `company` DOIT être commitée avant cette fixture
    """
    # ...
```

## 🚨 Pièges Courants

### 1. Objets expirés après rollback

**Problème** : Les objets SQLAlchemy peuvent être expirés après un rollback.

**Solution** : Toujours recharger depuis la DB après un rollback.

```python
db.session.rollback()
db.session.expire_all()
obj = db.session.query(MyModel).filter_by(id=obj.id).first()
```

### 2. Fixtures non commitées

**Problème** : Les fixtures qui utilisent `flush()` au lieu de `commit()` peuvent être expirées par `engine.run()`.

**Solution** : Utiliser `persisted_fixture()` ou appeler `commit()` explicitement.

### 3. Savepoints multiples

**Problème** : Créer des savepoints manuellement peut causer des problèmes d'isolation.

**Solution** : Utiliser `nested_savepoint()` pour gérer les savepoints imbriqués de manière sécurisée.

## 🔗 Découplage des Fixtures

Pour réduire les couplages entre fixtures, voir le [Guide de Découplage des Fixtures](../docs/FIXTURE_DECOUPLING.md).

**Principes** :

- ✅ Fixtures indépendantes (peuvent être utilisées seules)
- ✅ Dépendances optionnelles (paramètres avec valeur par défaut)
- ✅ Auto-création des dépendances si nécessaire
- ✅ Rétrocompatibilité maintenue

**Exemple** :

```python
@pytest.fixture
def drivers(db, company=None):  # ← Paramètre optionnel
    if company is None:
        company = CompanyFactory()  # Auto-création
        db.session.commit()
    return [DriverFactory(company=company) for _ in range(3)]
```

## 📚 Références

- [Guide de Découplage des Fixtures](../docs/FIXTURE_DECOUPLING.md) - Comment découpler les fixtures
- [Gestion des Sessions SQLAlchemy](../docs/SESSION_MANAGEMENT.md) - Guide complet de gestion des sessions (fixtures + code métier)
- [Tests de Non-Régression](./README_NON_REGRESSION.md) - Documentation des tests de non-régression
- [SQLAlchemy Session Management](https://docs.sqlalchemy.org/en/14/orm/session_transaction.html)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Factory Boy](https://factoryboy.readthedocs.io/)
