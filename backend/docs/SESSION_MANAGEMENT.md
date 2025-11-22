# 🔄 Gestion des Sessions SQLAlchemy - Guide Complet

Ce document décrit les bonnes pratiques pour gérer les sessions SQLAlchemy dans le projet ATMR, avec un focus sur l'isolation entre les fixtures de test et le code métier.

## 📋 Vue d'ensemble

Le projet utilise deux ensembles d'outils pour gérer les sessions SQLAlchemy :

1. **Pour le code métier** : Context managers dans `backend/services/db_context.py`
2. **Pour les tests** : Helpers dans `backend/tests/conftest.py`

## 🏗️ Architecture : Isolation Fixtures vs Code Métier

### Principe d'isolation

Les fixtures de test utilisent des **savepoints** (nested transactions) pour garantir l'isolation entre les tests, tandis que le code métier utilise des **transactions normales** avec gestion automatique des erreurs.

```
┌─────────────────────────────────────────────────────────┐
│                    FIXTURES (TESTS)                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Savepoint (nested transaction)                   │  │
│  │  - Isolation automatique entre tests              │  │
│  │  - Rollback automatique en fin de test            │  │
│  │  - Helpers: persisted_fixture(),                  │  │
│  │            ensure_committed(),                    │  │
│  │            nested_savepoint()                     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  CODE MÉTIER                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Transaction normale                              │  │
│  │  - Commit/rollback automatique                    │  │
│  │  - Gestion des erreurs                            │  │
│  │  - Context managers: db_transaction(),           │  │
│  │                     db_read_only(),              │  │
│  │                     db_batch_operation()         │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 🧪 Pour les Tests : Helpers dans `conftest.py`

### `persisted_fixture(db_session, factory_instance, model_class, **kwargs)`

Helper générique pour créer des fixtures persistées.

**Utilisation** :

```python
from tests.conftest import persisted_fixture
from tests.factories import CompanyFactory
from models import Company

@pytest.fixture
def company(db):
    return persisted_fixture(db, CompanyFactory(), Company)
```

**Avantages** :

- ✅ Commit automatique
- ✅ Rechargement depuis la DB pour garantir la persistance
- ✅ Vérification que l'objet existe

**Voir** : [README_FIXTURES.md](../tests/README_FIXTURES.md) pour plus de détails.

---

### `ensure_committed(db_session)`

Context manager pour garantir que tous les objets sont commités avant utilisation.

**Utilisation** :

```python
from tests.conftest import ensure_committed

def test_dispatch(db, company):
    with ensure_committed(db):
        # Tous les objets sont garantis commités ici
        result = engine.run(company_id=company.id, ...)
```

**Cas d'usage** :

- Forcer un commit explicite avant `engine.run()` (qui fait un rollback défensif)
- Garantir la persistance avant une opération critique

---

### `nested_savepoint(db_session)`

Context manager pour créer un savepoint imbriqué (nested transaction).

**Utilisation** :

```python
from tests.conftest import nested_savepoint

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

**⚠️ Attention** :

- Ne pas utiliser pour isoler des tests (utiliser la fixture `db` à la place)
- Utile pour tester des scénarios de rollback partiel dans un même test

---

## 💼 Pour le Code Métier : Context Managers dans `db_context.py`

### `db_transaction(auto_commit=True, auto_rollback=True, reraise=True)`

Context manager pour gérer proprement les transactions SQLAlchemy.

**Utilisation** :

```python
from services.db_context import db_transaction

# Simple transaction avec commit automatique
with db_transaction():
    invoice = Invoice(...)
    db.session.add(invoice)
    # Commit automatique à la fin

# Transaction sans commit automatique (commit manuel)
with db_transaction(auto_commit=False) as session:
    invoice = Invoice(...)
    session.add(invoice)
    session.flush()  # Pour obtenir l'ID sans committer
    # ... autres opérations
    session.commit()  # Commit manuel

# Transaction qui ne relève pas l'exception (logging seulement)
with db_transaction(reraise=False):
    risky_operation()
```

**Fonctionnalités** :

- ✅ Commit automatique si aucune exception
- ✅ Rollback automatique en cas d'exception
- ✅ Détection des tentatives d'écriture en mode read-only (chaos injector)
- ✅ Nettoyage automatique de la session (`session.remove()`)

**Paramètres** :

- `auto_commit` : Commit automatique si aucune exception (défaut: True)
- `auto_rollback` : Rollback automatique en cas d'exception (défaut: True)
- `reraise` : Re-lever l'exception après rollback (défaut: True)

---

### `db_read_only()`

Context manager pour les opérations de lecture seule.

**Utilisation** :

```python
from services.db_context import db_read_only

with db_read_only() as session:
    invoices = session.query(Invoice).filter_by(company_id=1).all()
    # Pas de commit (lecture seule)
```

**Fonctionnalités** :

- ✅ Pas de commit (lecture seule)
- ✅ Rollback automatique en cas d'erreur
- ✅ Nettoyage automatique de la session

---

### `db_batch_operation(batch_size=100, auto_commit_batch=True)`

Context manager pour les opérations par lot (batch) avec commits intermédiaires.

**Utilisation** :

```python
from services.db_context import db_batch_operation

with db_batch_operation(batch_size=100) as (session, commit_batch):
    for i, data in enumerate(large_dataset):
        invoice = Invoice(**data)
        session.add(invoice)

        if (i + 1) % 100 == 0:
            commit_batch()  # Commit intermédiaire tous les 100
```

**Fonctionnalités** :

- ✅ Commits intermédiaires pour éviter les transactions trop longues
- ✅ Commit final automatique si des opérations restantes
- ✅ Rollback automatique en cas d'erreur

**Paramètres** :

- `batch_size` : Nombre d'opérations avant un commit intermédiaire (défaut: 100)
- `auto_commit_batch` : Commit automatique à chaque lot (défaut: True)

---

## 🔄 Isolation entre Fixtures et Code Métier

### Principe

Les fixtures de test et le code métier utilisent des mécanismes différents pour gérer les sessions :

1. **Fixtures** : Utilisent des savepoints (nested transactions) pour l'isolation
2. **Code métier** : Utilise des transactions normales avec gestion automatique

### Exemple d'interaction

```python
# Dans un test
@pytest.fixture
def company(db):
    # Utilise un savepoint (via fixture db)
    return persisted_fixture(db, CompanyFactory(), Company)

def test_dispatch(company):
    # Le code métier (engine.run()) utilise une transaction normale
    # Les objets commités dans le savepoint sont visibles dans la transaction
    result = engine.run(company_id=company.id, ...)
```

### Points importants

- ✅ Les objets commités dans les fixtures (savepoints) sont visibles dans le code métier
- ✅ Le rollback défensif de `engine.run()` n'affecte pas les objets commités dans les fixtures
- ✅ Les fixtures garantissent l'isolation entre les tests via savepoints
- ✅ Le code métier gère ses propres transactions indépendamment

---

## 📝 Bonnes Pratiques

### 1. Dans les Tests

✅ **À faire** :

- Utiliser `persisted_fixture()` pour créer des fixtures persistées
- Utiliser `ensure_committed()` si nécessaire avant `engine.run()`
- Utiliser `nested_savepoint()` pour tester des scénarios de rollback partiel

❌ **À éviter** :

- Ne pas utiliser `db_transaction()` dans les tests (utiliser les fixtures)
- Ne pas créer de transactions manuelles dans les fixtures
- Ne pas réutiliser des objets expirés sans les recharger

### 2. Dans le Code Métier

✅ **À faire** :

- Utiliser `db_transaction()` pour toutes les opérations d'écriture
- Utiliser `db_read_only()` pour les opérations de lecture
- Utiliser `db_batch_operation()` pour les opérations par lot

❌ **À éviter** :

- Ne pas gérer manuellement les commits/rollbacks (utiliser les context managers)
- Ne pas oublier de nettoyer les sessions (`session.remove()` est géré automatiquement)
- Ne pas utiliser les helpers de test (`persisted_fixture()`, etc.) dans le code métier

### 3. Isolation

✅ **À faire** :

- Respecter la séparation entre fixtures (savepoints) et code métier (transactions)
- Documenter les interactions entre fixtures et code métier si nécessaire

❌ **À éviter** :

- Ne pas mélanger les mécanismes (savepoints dans le code métier, transactions dans les fixtures)
- Ne pas créer de dépendances circulaires entre fixtures et code métier

---

## 🔍 Dépannage

### Problème : Objets expirés après rollback

**Symptôme** : `DetachedInstanceError` ou objets avec valeurs None après rollback

**Solution** :

```python
# ❌ MAUVAIS
db.session.rollback()
obj = MyModel.query.get(id)  # Peut retourner None si expiré

# ✅ BON
db.session.rollback()
db.session.expire_all()
obj = db.session.query(MyModel).filter_by(id=id).first()  # Force un nouveau query
```

### Problème : Company introuvable dans engine.run()

**Symptôme** : `CompanyNotFoundError` ou `reason="company_not_found"` dans le résultat

**Solution** :

```python
# ✅ BON : Utiliser persisted_fixture() dans la fixture
@pytest.fixture
def company(db):
    return persisted_fixture(db, CompanyFactory(), Company)

# ✅ BON : Ou forcer un commit explicite
def test_dispatch(db, company):
    with ensure_committed(db):
        result = engine.run(company_id=company.id, ...)
```

### Problème : Fuites de connexions DB

**Symptôme** : Trop de connexions ouvertes, erreurs de pool

**Solution** :

- ✅ Utiliser les context managers (`db_transaction()`, etc.) qui appellent automatiquement `session.remove()`
- ✅ Ne pas créer de sessions manuelles sans les fermer
- ✅ Vérifier que les fixtures utilisent `db.session.remove()` (géré automatiquement par la fixture `db`)

---

## 📚 Références

- [Guide des Fixtures et Isolation](../tests/README_FIXTURES.md) - Documentation détaillée pour les tests
- [Tests de Non-Régression](../tests/README_NON_REGRESSION.md) - Tests critiques pour prévenir les régressions
- [SQLAlchemy Session Management](https://docs.sqlalchemy.org/en/14/orm/session_transaction.html) - Documentation officielle
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html) - Documentation officielle

---

**Note** : Ce document doit être mis à jour si de nouveaux context managers ou helpers sont ajoutés.
