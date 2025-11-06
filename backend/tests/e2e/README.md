# Tests E2E de Catastrophe et Chaos Engineering (D3)

Ce répertoire contient les tests end-to-end (E2E) pour valider la résilience du système face aux catastrophes.

## 📋 Table des matières

1. [Introduction](#introduction)
2. [Lancement des tests](#lancement-des-tests)
3. [Utilisation des injecteurs de chaos](#utilisation-des-injecteurs-de-chaos)
4. [Traffic Control (Optionnel)](#traffic-control-optionnel)
5. [Troubleshooting](#troubleshooting)

---

## Introduction

Les tests D3 valident que le système reste opérationnel même en cas de :

- **OSRM down** : Panne du service de routing
- **DB read-only** : Base de données en lecture seule
- **Pic de charge** : 500+ requêtes simultanées
- **Réseau flaky** : Latence élevée + erreurs réseau (30%)
- **Catastrophe combinée** : Plusieurs problèmes simultanés

### ⚠️ Sécurité

**NE JAMAIS activer le chaos en production !**

Les tests utilisent des injecteurs de chaos qui simulent des pannes. Ces injecteurs sont **désactivés par défaut** pour la sécurité.

---

## Lancement des tests

### Prérequis

- Python 3.11+
- pytest installé
- Base de données PostgreSQL accessible
- Services Docker démarrés (optionnel, pour tests complets)

### Commandes de base

```bash
# Lancer tous les tests E2E
pytest backend/tests/e2e/test_disaster_scenarios.py -v

# Lancer un test spécifique
pytest backend/tests/e2e/test_disaster_scenarios.py::TestDisasterScenarios::test_osrm_down_10_min -v

# Lancer avec logs détaillés
pytest backend/tests/e2e/test_disaster_scenarios.py -v -s --log-cli-level=INFO

# Lancer seulement les tests rapides (exclure pic de charge)
pytest backend/tests/e2e/test_disaster_scenarios.py -v -k "not pic_load"

# Lancer avec couverture
pytest backend/tests/e2e/test_disaster_scenarios.py -v --cov=backend/services --cov-report=html
```

### Variables d'environnement

Les tests utilisent les variables d'environnement suivantes (optionnelles) :

```bash
# Activer le chaos pour les tests (UNIQUEMENT en dev/test)
export CHAOS_ENABLED=true
export CHAOS_OSRM_DOWN=false
export CHAOS_DB_READ_ONLY=false

# Lancer les tests
pytest backend/tests/e2e/test_disaster_scenarios.py -v
```

**Note** : Les tests activent/désactivent automatiquement le chaos. Vous n'avez généralement pas besoin de définir ces variables manuellement.

---

## Utilisation des injecteurs de chaos

### Fixtures pytest disponibles

Le module `backend/tests/conftest.py` fournit plusieurs fixtures pour faciliter les tests :

#### 1. `chaos_injector`

Injecteur de chaos avec reset automatique après le test.

```python
def test_custom_chaos(chaos_injector):
    # Désactivé par défaut
    chaos_injector.enable()
    chaos_injector.set_latency(1000)  # 1 seconde
    chaos_injector.set_error_rate(0.1)  # 10% d'erreurs

    # ... votre test ...

    # Reset automatique à la fin
```

#### 2. `mock_osrm_down`

Active automatiquement OSRM down au début du test.

```python
def test_with_osrm_down(mock_osrm_down):
    # OSRM down est déjà activé automatiquement
    # Votre test peut utiliser OSRM (qui simulera une panne)

    from services.osrm_client import get_matrix
    result = get_matrix(...)  # Va lever ConnectionError

    # Restauration automatique à la fin
```

#### 3. `mock_db_read_only`

Active automatiquement DB read-only au début du test.

```python
def test_with_db_readonly(mock_db_read_only):
    # DB read-only est déjà activé automatiquement

    # Les lectures fonctionnent
    users = User.query.all()

    # Les écritures échouent avec RuntimeError
    try:
        user = User(...)
        db.session.add(user)
        db.session.commit()  # Lève RuntimeError
    except RuntimeError as e:
        assert "read-only" in str(e)

    # Restauration automatique à la fin
```

#### 4. `reset_chaos`

Reset complet du chaos injector (utilisé automatiquement par les autres fixtures).

```python
def test_with_manual_reset(reset_chaos):
    injector = reset_chaos

    injector.enable()
    injector.set_osrm_down(True)

    # ... test ...

    # Reset automatique dans finally
```

### Utilisation programmatique

Vous pouvez également utiliser l'injecteur directement dans vos tests :

```python
from chaos.injectors import get_chaos_injector

def test_manual_chaos():
    injector = get_chaos_injector()

    # Activer le chaos
    injector.enable()

    # Configurer
    injector.set_latency(500)  # 500ms
    injector.set_error_rate(0.2)  # 20% d'erreurs
    injector.set_osrm_down(True)
    injector.set_db_read_only(False)

    # ... votre test ...

    # Désactiver manuellement
    injector.disable()
    injector.set_osrm_down(False)
```

### Exemples avancés

#### Test avec latence progressive

```python
def test_latency_progression(chaos_injector):
    """Test avec latence qui augmente progressivement."""
    chaos_injector.enable()

    latencies = [100, 500, 1000, 2000]
    for latency_ms in latencies:
        chaos_injector.set_latency(latency_ms)

        # Mesurer la performance
        start = time.time()
        result = some_operation()
        duration = time.time() - start

        # Vérifier que la latence injectée est visible
        assert duration >= latency_ms / 1000.0
```

#### Test avec erreurs intermittentes

```python
def test_intermittent_errors(chaos_injector):
    """Test avec erreurs aléatoires."""
    chaos_injector.enable()
    chaos_injector.set_error_rate(0.3)  # 30% d'erreurs

    successes = 0
    failures = 0

    for _ in range(100):
        try:
            result = operation_that_may_fail()
            successes += 1
        except ConnectionError:
            failures += 1

    # Avec 30% d'erreurs, on devrait avoir ~30 erreurs
    assert 20 <= failures <= 40  # Tolérance
```

#### Test combinant plusieurs chaos

```python
def test_combined_chaos(chaos_injector):
    """Test avec plusieurs types de chaos simultanés."""
    chaos_injector.enable()

    # Activer plusieurs chaos
    chaos_injector.set_latency(1000)
    chaos_injector.set_error_rate(0.1)
    chaos_injector.set_osrm_down(False)  # Pas down, juste lent

    # Le système doit gérer tous ces problèmes
    result = complex_operation()

    assert result is not None
```

---

## Traffic Control (Optionnel)

**⚠️ Nécessite les privilèges root/sudo**

Le module `chaos.traffic_control` permet d'injecter de la latence et de la perte de paquets au niveau système (plus réaliste que l'injection Python).

### Prérequis

```bash
# Vérifier que TC est disponible
which tc

# Installer si nécessaire (Ubuntu/Debian)
sudo apt-get install iproute2
```

### Utilisation

```python
from chaos.traffic_control import TrafficControlManager

def test_with_system_latency():
    """Test avec latence injectée au niveau système."""
    tc = TrafficControlManager(interface="eth0")

    try:
        # Ajouter 500ms de latence
        success = tc.add_latency(500)
        if not success:
            pytest.skip("Requires root privileges")

        # Faire vos tests
        result = network_operation()

    finally:
        # IMPORTANT: Nettoyer les règles TC
        tc.clear()
```

### Commandes TC manuelles

Si vous préférez utiliser TC directement :

```bash
# Ajouter 500ms de latence sur eth0
sudo tc qdisc add dev eth0 root netem delay 500ms

# Ajouter 10% de perte de paquets
sudo tc qdisc add dev eth0 root netem loss 10%

# Voir les règles actives
sudo tc qdisc show dev eth0

# Supprimer toutes les règles
sudo tc qdisc del dev eth0 root
```

### Limitations

- **Nécessite root** : Pas disponible dans tous les environnements (CI/CD, conteneurs)
- **Interface spécifique** : Peut nécessiter d'adapter le nom de l'interface (`eth0`, `enp0s3`, etc.)
- **Impact système** : Affecte TOUT le trafic réseau sur l'interface

**Recommandation** : Utiliser l'injection Python (via `chaos_injector`) pour la plupart des tests, et TC uniquement pour les tests réseau très réalistes.

---

## Troubleshooting

### Les tests échouent avec "Chaos injector module not available"

**Cause** : Le module `chaos` n'est pas importable.

**Solution** :

```bash
# Vérifier que le PYTHONPATH inclut le répertoire backend
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"

# Ou lancer depuis le répertoire backend
cd backend
pytest tests/e2e/test_disaster_scenarios.py -v
```

### Les tests échouent avec "OSRM connection failed" mais OSRM fonctionne

**Cause** : Le chaos injector simule une panne OSRM.

**Solution** :

```python
# Vérifier l'état du chaos injector
from chaos.injectors import get_chaos_injector
injector = get_chaos_injector()
print(f"Enabled: {injector.enabled}, OSRM down: {injector.osrm_down}")

# Désactiver si nécessaire
injector.disable()
injector.set_osrm_down(False)
```

### Les tests sont trop lents

**Cause** : Les tests E2E peuvent être longs (surtout le pic de charge avec 500 requêtes).

**Solutions** :

```bash
# Lancer seulement les tests rapides
pytest backend/tests/e2e/test_disaster_scenarios.py -v -k "not pic_load and not combined"

# Réduire le nombre de requêtes pour le pic de charge
# Modifier PIC_LOAD_REQUESTS dans test_disaster_scenarios.py
```

### Le test `test_db_read_only` échoue avec des erreurs SQL

**Cause** : Le middleware ou `db_transaction` ne détecte pas correctement le mode read-only.

**Solution** :

1. Vérifier que le chaos est activé :

```python
from chaos.injectors import get_chaos_injector
injector = get_chaos_injector()
assert injector.enabled
assert injector.db_read_only
```

2. Vérifier les logs pour voir si le middleware bloque correctement :

```bash
pytest backend/tests/e2e/test_disaster_scenarios.py::TestDisasterScenarios::test_db_read_only -v -s --log-cli-level=WARNING
```

### Le fallback haversine n'est pas détecté

**Cause** : Le système utilise peut-être le cache OSRM au lieu du fallback haversine.

**Solution** :

- Vider le cache Redis avant le test
- Vérifier les logs avec `--log-cli-level=INFO` pour voir les messages de fallback
- Le test accepte aussi que le système fonctionne via cache (pas seulement haversine)

### Erreur "Requires root privileges" avec Traffic Control

**Cause** : TC nécessite les privilèges root/sudo.

**Solutions** :

1. Utiliser l'injection Python au lieu de TC (recommandé)
2. Lancer les tests avec sudo (non recommandé en CI/CD)
3. Ignorer les tests qui nécessitent TC avec `pytest.skip()`

### Les fixtures ne nettoient pas correctement le chaos

**Cause** : Une exception empêche le nettoyage dans `finally`.

**Solution** :

- Vérifier que le chaos est bien désactivé après chaque test :

```python
# Ajouter dans conftest.py ou vos tests
@pytest.fixture(autouse=True)
def verify_chaos_disabled():
    yield
    # Après chaque test, vérifier que chaos est désactivé
    from chaos.injectors import get_chaos_injector
    injector = get_chaos_injector()
    assert not injector.enabled, "Chaos should be disabled after test"
```

### Les tests passent localement mais échouent en CI/CD

**Causes possibles** :

1. **Variables d'environnement différentes** : Vérifier `.env` en CI
2. **Timing différent** : Les tests peuvent être plus lents en CI
3. **Services non disponibles** : PostgreSQL, Redis, OSRM doivent être démarrés

**Solutions** :

```yaml
# Exemple GitHub Actions
- name: Start services
  run: |
    docker-compose up -d postgres redis osrm
    sleep 10  # Attendre que les services soient prêts

- name: Run E2E tests
  run: |
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"
    pytest backend/tests/e2e/test_disaster_scenarios.py -v --maxfail=1
```

---

## Ressources supplémentaires

- **Runbook** : `backend/RUNBOOK.md` - Procédures opérationnelles pour gérer les catastrophes
- **TODO D3** : `backend/tests/e2e/TODO_D3.md` - Liste complète des tâches D3
- **Code des tests** : `backend/tests/e2e/test_disaster_scenarios.py`
- **Injecteurs de chaos** : `backend/chaos/injectors.py`

---

_Dernière mise à jour: 2025-10-28_10:40
