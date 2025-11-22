# 🔗 Guide de Découplage des Fixtures

Ce document explique comment découpler les fixtures pour réduire les dépendances et améliorer la maintenabilité des tests.

## 📋 Problème : Couplages Actuels

### Dépendances en Chaîne

Actuellement, certaines fixtures ont des dépendances en chaîne :

```
sample_user → sample_company → sample_client
company → drivers
company → bookings
```

**Problèmes** :

- ⚠️ Les fixtures dépendent de l'ordre d'exécution
- ⚠️ Impossible d'utiliser `drivers` sans `company`
- ⚠️ Modification d'une fixture peut casser les autres
- ⚠️ Tests plus difficiles à comprendre et maintenir

## ✅ Solution : Fixtures Indépendantes

### Principe : Auto-création des Dépendances

Au lieu de dépendre d'autres fixtures, chaque fixture peut créer ses propres dépendances si nécessaire :

```python
# ❌ AVANT : Dépendance explicite
@pytest.fixture
def drivers(db, company):
    """Dépend de company."""
    return [DriverFactory(company=company) for _ in range(3)]

# ✅ APRÈS : Auto-création
@pytest.fixture
def drivers(db):
    """Crée sa propre company si nécessaire."""
    company = CompanyFactory()
    db.session.commit()
    return [DriverFactory(company=company) for _ in range(3)]
```

### Avantages

- ✅ **Indépendance** : Chaque fixture peut être utilisée seule
- ✅ **Flexibilité** : Possibilité de passer une company existante si nécessaire
- ✅ **Maintenabilité** : Modification d'une fixture n'affecte pas les autres
- ✅ **Clarté** : Les dépendances sont explicites dans le code

## 🔧 Patterns de Découplage

### Pattern 1 : Fixture avec Paramètre Optionnel

```python
@pytest.fixture
def drivers(db, company=None):
    """Crée des drivers, avec company optionnelle."""
    if company is None:
        # Auto-création si non fournie
        company = CompanyFactory()
        db.session.commit()

    drivers_list = [DriverFactory(company=company) for _ in range(3)]
    db.session.commit()
    return drivers_list
```

**Utilisation** :

```python
# Utilisation indépendante
def test_drivers_only(drivers):
    # company créée automatiquement
    pass

# Utilisation avec company existante
def test_with_company(company, drivers):
    # company passée explicitement
    pass
```

### Pattern 2 : Fixture avec Factory Function

```python
def create_drivers_for_company(db, company, count=3):
    """Factory function pour créer des drivers."""
    drivers_list = [DriverFactory(company=company) for _ in range(count)]
    db.session.commit()
    return drivers_list

@pytest.fixture
def drivers(db):
    """Crée des drivers avec company auto-créée."""
    company = CompanyFactory()
    db.session.commit()
    return create_drivers_for_company(db, company)
```

**Utilisation** :

```python
# Utilisation de la fixture
def test_drivers(drivers):
    pass

# Utilisation directe de la factory
def test_custom_drivers(db, company):
    drivers = create_drivers_for_company(db, company, count=5)
    pass
```

### Pattern 3 : Fixture avec Scope et Cache

```python
@pytest.fixture(scope="function")
def company(db):
    """Company indépendante, créée à la demande."""
    company = CompanyFactory()
    db.session.commit()
    return company

@pytest.fixture
def drivers(db):
    """Drivers indépendants, créent leur propre company."""
    company = CompanyFactory()
    db.session.commit()
    return [DriverFactory(company=company) for _ in range(3)]
```

**Avantages** :

- ✅ Chaque fixture est indépendante
- ✅ Pas de dépendance explicite
- ✅ Isolation garantie par les savepoints

## 📝 Migration Guide

### Étape 1 : Identifier les Dépendances

```python
# Identifier les dépendances actuelles
@pytest.fixture
def drivers(db, company):  # ← Dépend de company
    ...
```

### Étape 2 : Rendre la Dépendance Optionnelle

```python
@pytest.fixture
def drivers(db, company=None):  # ← Optionnelle
    if company is None:
        company = CompanyFactory()
        db.session.commit()
    ...
```

### Étape 3 : Documenter le Comportement

```python
@pytest.fixture
def drivers(db, company=None):
    """Crée des drivers pour les tests.

    Args:
        db: Session SQLAlchemy (requis)
        company: Company existante (optionnel, créée si None)

    Returns:
        Liste de drivers persistés
    """
    ...
```

## 🎯 Exemples Concrets

### Exemple 1 : Découpler `drivers` de `company`

**Avant** :

```python
@pytest.fixture
def drivers(db, company):
    """Dépend de company."""
    return [DriverFactory(company=company) for _ in range(3)]
```

**Après** :

```python
@pytest.fixture
def drivers(db, company=None):
    """Crée des drivers, avec company optionnelle.

    Si company n'est pas fournie, une company est créée automatiquement.
    """
    if company is None:
        company = CompanyFactory()
        db.session.commit()

    drivers_list = [DriverFactory(company=company) for _ in range(3)]
    db.session.commit()
    return drivers_list
```

### Exemple 2 : Découpler `bookings` de `company`

**Avant** :

```python
@pytest.fixture
def bookings(db, company):
    """Dépend de company."""
    bookings_list = []
    for i in range(5):
        booking = BookingFactory(company=company)
        bookings_list.append(booking)
    db.session.commit()
    return bookings_list
```

**Après** :

```python
@pytest.fixture
def bookings(db, company=None):
    """Crée des bookings, avec company optionnelle.

    Si company n'est pas fournie, une company est créée automatiquement.
    """
    if company is None:
        company = CompanyFactory()
        db.session.commit()

    bookings_list = []
    for i in range(5):
        booking = BookingFactory(company=company)
        bookings_list.append(booking)
    db.session.commit()
    return bookings_list
```

## ⚠️ Points d'Attention

### 1. Isolation des Tests

Même avec des fixtures découplées, l'isolation est garantie par les savepoints :

```python
def test_example(db, drivers):
    # drivers crée sa propre company
    # Le savepoint garantit l'isolation
    pass
```

### 2. Performance

Les fixtures découplées peuvent créer plus d'objets (une company par fixture), mais :

- ✅ L'isolation est meilleure
- ✅ Les tests sont plus maintenables
- ✅ L'impact sur la performance est négligeable (savepoints rapides)

### 3. Rétrocompatibilité

Pour maintenir la rétrocompatibilité, garder les paramètres optionnels :

```python
@pytest.fixture
def drivers(db, company=None):  # ← Paramètre optionnel
    # Compatible avec l'ancien usage (company passée)
    # Et avec le nouvel usage (company=None, auto-création)
    ...
```

## 📊 État Actuel vs Cible

### État Actuel

```
company → drivers
company → bookings
sample_user → sample_company → sample_client
```

### État Cible

```
company (indépendant)
drivers (indépendant, company optionnelle)
bookings (indépendant, company optionnelle)
sample_user (indépendant)
sample_company (indépendant, sample_user optionnel)
sample_client (indépendant, sample_company optionnelle)
```

## 🔗 Références

- [Guide des Fixtures et Isolation](../tests/README_FIXTURES.md) - Documentation des fixtures
- [Gestion des Sessions SQLAlchemy](./SESSION_MANAGEMENT.md) - Guide complet de gestion des sessions
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html) - Documentation officielle

---

**Note** : Cette migration peut être effectuée progressivement, fixture par fixture, en maintenant la rétrocompatibilité.
