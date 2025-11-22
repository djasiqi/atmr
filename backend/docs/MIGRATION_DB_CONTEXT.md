# 🔄 Guide de Migration vers `db_context.py`

Ce guide explique comment migrer le code existant pour utiliser les context managers de `db_context.py` au lieu de gérer manuellement les sessions SQLAlchemy.

## 🎯 Objectif

Standardiser la gestion des sessions SQLAlchemy en utilisant les context managers de `db_context.py` pour :

- ✅ Réduire la duplication de code (try/except/finally)
- ✅ Garantir le nettoyage automatique des sessions
- ✅ Améliorer la gestion d'erreurs
- ✅ Faciliter le monitoring (métriques automatiques)

## 📋 Patterns à Migrer

### Pattern 1 : Transaction Simple avec Commit Automatique

**❌ AVANT** :

```python
try:
    invoice = Invoice(...)
    db.session.add(invoice)
    db.session.commit()
except Exception as e:
    db.session.rollback()
    raise
finally:
    db.session.remove()
```

**✅ APRÈS** :

```python
from services.db_context import db_transaction

with db_transaction():
    invoice = Invoice(...)
    db.session.add(invoice)
    # Commit automatique à la fin
```

---

### Pattern 2 : Transaction avec Commit Manuel

**❌ AVANT** :

```python
try:
    invoice = Invoice(...)
    db.session.add(invoice)
    db.session.flush()  # Pour obtenir l'ID
    # ... autres opérations
    db.session.commit()
except Exception as e:
    db.session.rollback()
    raise
finally:
    db.session.remove()
```

**✅ APRÈS** :

```python
from services.db_context import db_transaction

with db_transaction(auto_commit=False) as session:
    invoice = Invoice(...)
    session.add(invoice)
    session.flush()  # Pour obtenir l'ID
    # ... autres opérations
    session.commit()  # Commit manuel
```

---

### Pattern 3 : Opérations de Lecture Seule

**❌ AVANT** :

```python
try:
    invoices = db.session.query(Invoice).filter_by(company_id=1).all()
except Exception as e:
    db.session.rollback()
    raise
finally:
    db.session.remove()
```

**✅ APRÈS** :

```python
from services.db_context import db_read_only

with db_read_only() as session:
    invoices = session.query(Invoice).filter_by(company_id=1).all()
    # Pas de commit (lecture seule)
```

---

### Pattern 4 : Opérations par Lot (Batch)

**❌ AVANT** :

```python
try:
    for i, data in enumerate(large_dataset):
        invoice = Invoice(**data)
        db.session.add(invoice)
        if (i + 1) % 100 == 0:
            db.session.commit()
    db.session.commit()  # Commit final
except Exception as e:
    db.session.rollback()
    raise
finally:
    db.session.remove()
```

**✅ APRÈS** :

```python
from services.db_context import db_batch_operation

with db_batch_operation(batch_size=100) as (session, commit_batch):
    for i, data in enumerate(large_dataset):
        invoice = Invoice(**data)
        session.add(invoice)
        if (i + 1) % 100 == 0:
            commit_batch()  # Commit intermédiaire
    # Commit final automatique si des opérations restantes
```

---

### Pattern 5 : Transaction qui Ne Relève Pas l'Exception

**❌ AVANT** :

```python
try:
    risky_operation()
except Exception as e:
    db.session.rollback()
    logger.error("Operation failed: %s", e)
    # Ne pas relever l'exception
finally:
    db.session.remove()
```

**✅ APRÈS** :

```python
from services.db_context import db_transaction

with db_transaction(reraise=False):
    risky_operation()
    # Logging automatique en cas d'erreur, pas d'exception levée
```

---

## 🔍 Identification du Code à Migrer

### Fichiers avec Usage Direct de `db.session`

Les fichiers suivants utilisent directement `db.session.commit()` ou `db.session.rollback()` :

**Services** :

- `backend/services/unified_dispatch/engine.py` - Utilise `db.session.rollback()` et `db.session.commit()`
- `backend/services/unified_dispatch/apply.py` - Utilise `db.session.flush()` et `db.session.commit()`

**Routes** :

- `backend/routes/companies.py` - Utilise `db.session.add()`, `db.session.commit()`, `db.session.rollback()`

### Stratégie de Migration

1. **Priorité 1** : Code critique (dispatch, apply)

   - `backend/services/unified_dispatch/engine.py`
   - `backend/services/unified_dispatch/apply.py`

2. **Priorité 2** : Routes API

   - `backend/routes/companies.py`
   - Autres routes avec usage direct

3. **Priorité 3** : Code moins critique
   - Services auxiliaires
   - Scripts

---

## 📝 Exemple de Migration Complète

### Exemple : Migration d'une Route API

**❌ AVANT** (`backend/routes/companies.py`) :

```python
@companies_ns.route("/<int:company_id>")
class CompanyResource(Resource):
    @jwt_required()
    def put(self, company_id):
        try:
            company = Company.query.get(company_id)
            if not company:
                return {"error": "Company not found"}, 404

            # Mise à jour
            company.name = request.json.get("name", company.name)
            db.session.commit()
            return company.serialize, 200
        except Exception as e:
            db.session.rollback()
            logger.error("Error updating company: %s", e)
            return {"error": str(e)}, 500
        finally:
            db.session.remove()
```

**✅ APRÈS** :

```python
from services.db_context import db_transaction

@companies_ns.route("/<int:company_id>")
class CompanyResource(Resource):
    @jwt_required()
    def put(self, company_id):
        with db_transaction():
            company = Company.query.get(company_id)
            if not company:
                return {"error": "Company not found"}, 404

            # Mise à jour
            company.name = request.json.get("name", company.name)
            # Commit automatique à la fin du context manager
            return company.serialize, 200
        # Rollback et remove automatiques en cas d'exception
```

---

## ⚠️ Points d'Attention

### 1. Gestion des Exceptions

Les context managers gèrent automatiquement les exceptions, mais vous pouvez toujours utiliser `reraise=False` si nécessaire :

```python
with db_transaction(reraise=False):
    risky_operation()
    # Si une exception survient, elle est loggée mais pas relevée
```

### 2. Commit Manuel

Si vous avez besoin d'un commit manuel (pour obtenir un ID avant d'autres opérations), utilisez `auto_commit=False` :

```python
with db_transaction(auto_commit=False) as session:
    invoice = Invoice(...)
    session.add(invoice)
    session.flush()  # Obtenir l'ID
    # ... autres opérations
    session.commit()  # Commit manuel
```

### 3. Nettoyage de Session

Les context managers appellent automatiquement `session.remove()` dans le `finally`, donc vous n'avez plus besoin de le faire manuellement.

### 4. Mode Read-Only

Les context managers détectent automatiquement le mode read-only (chaos injector) et bloquent les écritures si nécessaire.

---

## 📊 Monitoring et Métriques

Après migration, les métriques suivantes sont automatiquement trackées :

- `db_transaction_total{operation="commit"}` - Nombre de commits
- `db_transaction_total{operation="rollback"}` - Nombre de rollbacks
- `db_transaction_duration_seconds{operation="commit"}` - Durée des commits
- `db_context_manager_usage_total{manager_type="db_transaction"}` - Utilisation des context managers
- `db_direct_session_usage_total{operation="commit"}` - Usage direct (à réduire)

**Voir** : `backend/services/db_session_metrics.py` pour plus de détails.

---

## ✅ Checklist de Migration

Pour chaque fichier à migrer :

- [ ] Identifier tous les usages de `db.session.commit()`, `db.session.rollback()`, `db.session.remove()`
- [ ] Remplacer par le context manager approprié (`db_transaction()`, `db_read_only()`, `db_batch_operation()`)
- [ ] Supprimer les blocs `try/except/finally` redondants
- [ ] Tester que le comportement est identique
- [ ] Vérifier que les métriques sont trackées correctement
- [ ] Mettre à jour la documentation si nécessaire

---

## 🔗 Références

- [Gestion des Sessions SQLAlchemy](./SESSION_MANAGEMENT.md) - Guide complet
- [Guide des Fixtures et Isolation](../tests/README_FIXTURES.md) - Pour les tests
- [SQLAlchemy Session Management](https://docs.sqlalchemy.org/en/14/orm/session_transaction.html) - Documentation officielle

---

**Note** : Cette migration peut être effectuée progressivement, fichier par fichier. Les métriques permettent de suivre l'avancement de la migration.
