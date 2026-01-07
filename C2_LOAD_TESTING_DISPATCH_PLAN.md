# ⚡ C2 - Load Testing Dispatch - Plan d'Implémentation

**Date :** 7 janvier 2025 - 22h45  
**Référence :** `AUDIT_TECHNIQUE_COMPLET_2025.md` (Section C2, lignes 1532-1547)  
**Objectif :** Valider performance du système de dispatch sous charge  
**Durée estimée :** 1 semaine (setup + run + analyse)

---

## 🎯 Objectifs

**Valider la performance du dispatch sous charge réelle :**

1. ✅ Identifier les goulots d'étranglement
2. ✅ Valider la scalabilité (100 bookings × 50 drivers = 5000 paires)
3. ✅ Mesurer les temps de réponse sous charge
4. ✅ Tester la résilience avec OSRM lent
5. ✅ Tester le parallélisme multi-entreprises

---

## 📊 Scénarios de Test

### Scénario 1 : Charge Standard (P0)

**Description :** 100 bookings + 50 drivers (matrices 100×50)

**Objectifs :**

- Temps de calcul matrices : <30s
- Temps total dispatch : <60s
- Mémoire : <4GB
- CPU : <80% utilisation

**Charge simulée :**

```
1 entreprise
100 réservations actives
50 chauffeurs disponibles
= 5000 combinaisons à évaluer
```

---

### Scénario 2 : Parallélisme Multi-Entreprises (P0)

**Description :** 10 entreprises en parallèle

**Objectifs :**

- Toutes les entreprises dispatches en <90s
- Pas de deadlocks base de données
- Isolation correcte des données
- Redis cache efficace

**Charge simulée :**

```
10 entreprises × 20 bookings × 10 drivers = 10 dispatch simultanés
```

---

### Scénario 3 : OSRM Lent (P1)

**Description :** Latence OSRM 500ms (réseau lent/chargé)

**Objectifs :**

- Dispatch fonctionne malgré latence
- Timeout OSRM configuré correctement
- Fallback Haversine si nécessaire
- Temps total dispatch : <120s

**Charge simulée :**

```
1 entreprise
50 bookings × 20 drivers avec OSRM @ 500ms latency
```

---

## 🛠️ Outil : Locust vs K6

### Comparaison

| Critère                  | Locust        | K6         | Décision   |
| ------------------------ | ------------- | ---------- | ---------- |
| **Langage**              | Python ✅     | JavaScript | **Locust** |
| **Courbe apprentissage** | Facile        | Moyenne    | **Locust** |
| **Intégration backend**  | Native Python | Via HTTP   | **Locust** |
| **Rapport**              | Web UI ✅     | CLI + JSON | **Locust** |
| **Communauté**           | Large         | Large      | Égalité    |

**Décision :** **Locust** ✅ (meilleure intégration avec backend Python)

---

## 📝 Plan d'Implémentation (1 semaine)

### Jour 1 : Setup Locust

**Objectifs :**

- ✅ Installer Locust
- ✅ Créer structure tests
- ✅ Test Hello World

**Actions :**

```bash
cd backend
pip install locust
mkdir -p tests/load_testing
touch tests/load_testing/locustfile.py
```

---

### Jour 2 : Scénario 1 (Charge Standard)

**Objectifs :**

- ✅ Implémenter test 100 bookings × 50 drivers
- ✅ Mesurer temps dispatch
- ✅ Capturer métriques

**Fichier :** `tests/load_testing/dispatch_load_test.py`

**Code :**

```python
from locust import HttpUser, task, between
import json

class DispatchLoadTest(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Login et setup initial"""
        self.client.post("/api/login", json={
            "email": "admin@test.com",
            "password": "test123"
        })

    @task
    def dispatch_large_set(self):
        """Test dispatch 100 bookings × 50 drivers"""
        response = self.client.post(
            "/api/dispatch/run",
            json={
                "company_id": 1,
                "date": "2025-01-08",
                "mode": "optimization"
            }
        )

        if response.status_code == 200:
            data = response.json()
            # Log métriques
            print(f"Dispatch time: {data.get('duration_seconds')}s")
            print(f"Assignments: {data.get('num_assignments')}")
```

---

### Jour 3 : Scénario 2 (Multi-Entreprises)

**Objectifs :**

- ✅ Implémenter test 10 entreprises parallèles
- ✅ Valider isolation données
- ✅ Mesurer contention DB

**Code :**

```python
class MultiCompanyDispatchTest(HttpUser):
    wait_time = between(0, 1)  # Parallélisme agressif

    @task
    def dispatch_company(self):
        """Dispatch pour une entreprise random"""
        company_id = random.randint(1, 10)
        response = self.client.post(
            f"/api/dispatch/run",
            json={
                "company_id": company_id,
                "date": "2025-01-08"
            }
        )
```

---

### Jour 4 : Scénario 3 (OSRM Lent)

**Objectifs :**

- ✅ Simuler latence OSRM 500ms
- ✅ Valider timeouts
- ✅ Tester fallback Haversine

**Setup :**

```python
# Configuration Docker pour OSRM lent
# docker-compose.yml
osrm-slow:
  image: osrm/osrm-backend
  command: >
    sh -c "tc qdisc add dev eth0 root netem delay 500ms &&
           osrm-routed --algorithm mld /data/map.osrm"
```

**Test :**

```python
class SlowOSRMDispatchTest(HttpUser):
    host = "http://localhost:5000"

    @task
    def dispatch_slow_osrm(self):
        """Test dispatch avec OSRM lent (500ms)"""
        start = time.time()
        response = self.client.post(
            "/api/dispatch/run",
            json={
                "company_id": 1,
                "date": "2025-01-08",
                "osrm_timeout": 10  # 10s timeout
            }
        )
        duration = time.time() - start

        # Vérifier que le dispatch a réussi malgré latence
        assert response.status_code == 200
        assert duration < 120  # <2min acceptable
```

---

### Jour 5 : Data Fixtures & Validation

**Objectifs :**

- ✅ Créer fixtures pour tests
- ✅ 100 bookings réalistes
- ✅ 50 drivers réalistes
- ✅ Validation données avant tests

**Script :**

```python
# tests/load_testing/fixtures/create_test_data.py
def create_test_bookings(num=100):
    """Créer 100 réservations pour tests de charge"""
    for i in range(num):
        booking = Booking(
            pickup_address=f"Test Pickup {i}",
            dropoff_address=f"Test Dropoff {i}",
            pickup_time=datetime.now() + timedelta(hours=i),
            company_id=1,
            status="pending"
        )
        db.session.add(booking)
    db.session.commit()

def create_test_drivers(num=50):
    """Créer 50 chauffeurs pour tests de charge"""
    for i in range(num):
        driver = Driver(
            name=f"Test Driver {i}",
            company_id=1,
            status="available",
            latitude=46.5 + (i * 0.01),
            longitude=6.5 + (i * 0.01)
        )
        db.session.add(driver)
    db.session.commit()
```

---

### Jour 6 : Exécution Tests & Collecte Métriques

**Objectifs :**

- ✅ Exécuter tous les scénarios
- ✅ Collecter métriques détaillées
- ✅ Identifier goulots d'étranglement

**Commandes :**

```bash
# Test 1 : Charge standard (10 users, 2 min)
locust -f tests/load_testing/dispatch_load_test.py \
       --headless -u 10 -r 2 -t 2m \
       --html reports/scenario1.html

# Test 2 : Multi-entreprises (20 users, 5 min)
locust -f tests/load_testing/multi_company_test.py \
       --headless -u 20 -r 5 -t 5m \
       --html reports/scenario2.html

# Test 3 : OSRM lent (5 users, 3 min)
locust -f tests/load_testing/slow_osrm_test.py \
       --headless -u 5 -r 1 -t 3m \
       --html reports/scenario3.html
```

**Métriques à capturer :**

- Temps de réponse (p50, p95, p99)
- Throughput (req/s)
- Taux d'erreur
- Utilisation CPU/RAM
- Temps DB queries
- Temps OSRM calls

---

### Jour 7 : Analyse & Rapport Final

**Objectifs :**

- ✅ Analyser résultats
- ✅ Identifier optimisations
- ✅ Créer rapport final

**Livrables :**

- `C2_LOAD_TESTING_RESULTS.md` (rapport détaillé)
- Graphiques temps de réponse
- Recommandations optimisation

---

## 📊 Métriques de Succès

| Métrique                                 | Objectif | Critique |
| ---------------------------------------- | -------- | -------- |
| **Scénario 1 : Temps dispatch**          | <60s     | <90s     |
| **Scénario 1 : Mémoire**                 | <4GB     | <6GB     |
| **Scénario 1 : Taux erreur**             | 0%       | <1%      |
| **Scénario 2 : Temps par entreprise**    | <90s     | <120s    |
| **Scénario 2 : Deadlocks DB**            | 0        | 0        |
| **Scénario 3 : Dispatch malgré latence** | 100%     | >95%     |
| **Scénario 3 : Temps total**             | <120s    | <180s    |

---

## 🚧 Risques Identifiés

| Risque                  | Probabilité | Impact | Mitigation                                  |
| ----------------------- | ----------- | ------ | ------------------------------------------- |
| Base de données saturée | Moyenne     | Élevé  | Optimiser indices, connection pooling       |
| Mémoire insuffisante    | Moyenne     | Élevé  | Augmenter limite Docker, optimiser matrices |
| OSRM timeout            | Faible      | Moyen  | Fallback Haversine déjà implémenté          |
| Redis cache inefficace  | Faible      | Moyen  | Analyser hit ratio, ajuster TTL             |

---

## 📦 Dépendances Requises

```bash
# Installation Locust
pip install locust

# Installation monitoring (optionnel)
pip install py-spy memory_profiler

# Installation Docker pour OSRM slow
docker-compose -f docker-compose.load-test.yml up -d
```

---

## 🔧 Configuration Docker pour Tests

**Fichier :** `docker-compose.load-test.yml`

```yaml
version: "3.8"

services:
  # OSRM normal (baseline)
  osrm-fast:
    image: osrm/osrm-backend
    ports:
      - "5000:5000"
    volumes:
      - ./osrm/data:/data

  # OSRM lent (500ms latency simulée)
  osrm-slow:
    image: osrm/osrm-backend
    ports:
      - "5001:5000"
    volumes:
      - ./osrm/data:/data
    cap_add:
      - NET_ADMIN
    command: >
      sh -c "tc qdisc add dev eth0 root netem delay 500ms && 
             osrm-routed --algorithm mld /data/map.osrm"

  # Base de données dédiée tests
  postgres-test:
    image: postgres:16
    environment:
      POSTGRES_DB: atmr_load_test
      POSTGRES_USER: atmr
      POSTGRES_PASSWORD: atmr
    ports:
      - "5433:5432"
    volumes:
      - test-data:/var/lib/postgresql/data

volumes:
  test-data:
```

---

## 📝 Structure Fichiers

```
backend/tests/load_testing/
├── __init__.py
├── locustfile.py                 # Entry point
├── scenarios/
│   ├── __init__.py
│   ├── dispatch_standard.py      # Scénario 1
│   ├── dispatch_multi_company.py # Scénario 2
│   └── dispatch_slow_osrm.py     # Scénario 3
├── fixtures/
│   ├── __init__.py
│   ├── create_test_data.py       # Génération fixtures
│   └── cleanup.py                # Nettoyage après tests
├── reports/                      # Rapports HTML Locust
│   ├── scenario1.html
│   ├── scenario2.html
│   └── scenario3.html
└── utils/
    ├── __init__.py
    ├── metrics.py                # Collecte métriques custom
    └── monitoring.py             # Monitoring CPU/RAM
```

---

## 🎯 Prochaines Actions Immédiates

1. ✅ **Créer plan C2** (ce document)
2. 🔲 **Installer Locust** (`pip install locust`)
3. 🔲 **Créer structure tests** (`tests/load_testing/`)
4. 🔲 **Démarrer Jour 1** (Setup Locust)

---

**Date création :** 7 janvier 2025 - 23h00  
**Status :** 🔵 **PLANIFIÉ** - Prêt à démarrer  
**Durée estimée :** 7 jours
