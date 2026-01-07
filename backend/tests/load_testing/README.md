# 🔥 Tests de Charge (Load Testing) - Locust

Ce dossier contient les tests de charge pour valider la performance et la résilience du système de dispatch ATMR.

## 📋 Scénarios Disponibles

### Scénario 1 : Charge Standard (`dispatch_load_test.py`)

**Objectif :** Valider performance avec charge importante

- **Données :** 100 bookings × 50 drivers
- **Matrices :** 5000 éléments (calcul distances)
- **Optimisation :** OR-Tools (MIP)
- **Durée recommandée :** 10-15 min
- **Users recommandés :** 1-10

**SLO :**
- Dispatch duration : < 60s
- Taux assignation : > 80%
- Success rate : > 95%

### Scénario 2 : Multi-Entreprises (`multi_company_test.py`)

**Objectif :** Valider isolation et contention

- **Entreprises :** 10 en parallèle
- **Validation :** Isolation données, locks Redis
- **Contention :** DB queries simultanées
- **Durée recommandée :** 10 min
- **Users recommandés :** 10-20

**SLO :**
- Pas de leak données entre entreprises
- Locks Redis fonctionnels
- Pas de deadlocks DB
- Performance stable

### Scénario 3 : OSRM Lent (`slow_osrm_test.py`)

**Objectif :** Valider résilience avec OSRM dégradé

- **Latence OSRM :** 500ms (simulée)
- **Validation :** Timeouts, fallback, circuit breaker
- **Cache :** Efficacité hit rate > 80%
- **Durée recommandée :** 10-15 min
- **Users recommandés :** 5-10

**SLO :**
- Dispatch duration : < 120s (malgré latence)
- Fallback Haversine fonctionnel
- Circuit breaker actif si nécessaire
- Success rate : > 90%

---

## 🚀 Installation

### Prérequis

- Python 3.11+
- Docker + Docker Compose
- Backend ATMR running

### Installation Locust

```bash
cd backend
pip install locust
```

**Vérification :**
```bash
locust --version
# locust 2.x.x
```

---

## 📖 Usage

### Mode Web UI (Recommandé pour débuter)

```bash
# Scénario 1
locust -f tests/load_testing/dispatch_load_test.py --host=http://localhost:5000

# Ouvrir http://localhost:8089
# Configurer : Users=5, Spawn rate=1, Run time=5m
# Cliquer "Start swarming"
```

### Mode Headless (Ligne de commande)

```bash
# Scénario 1 : Test léger
locust -f tests/load_testing/dispatch_load_test.py \
    --host=http://localhost:5000 \
    --users=5 \
    --spawn-rate=1 \
    --run-time=10m \
    --headless \
    --csv=results_scenario1

# Scénario 2 : Multi-entreprises
locust -f tests/load_testing/multi_company_test.py \
    --host=http://localhost:5000 \
    --users=10 \
    --spawn-rate=10 \
    --run-time=10m \
    --headless \
    --csv=results_scenario2

# Scénario 3 : OSRM lent
locust -f tests/load_testing/slow_osrm_test.py \
    --host=http://localhost:5000 \
    --users=5 \
    --spawn-rate=1 \
    --run-time=10m \
    --headless \
    --csv=results_scenario3
```

### Mode Distribué (Master/Workers)

```bash
# Terminal 1 : Master
locust -f tests/load_testing/dispatch_load_test.py \
    --master \
    --host=http://localhost:5000 \
    --expect-workers=4

# Terminaux 2-5 : Workers
locust -f tests/load_testing/dispatch_load_test.py \
    --worker \
    --master-host=localhost
```

---

## 📊 Résultats et Métriques

### Fichiers générés (mode `--csv`)

```
results_scenario1_stats.csv         # Statistiques globales
results_scenario1_stats_history.csv # Historique temporel
results_scenario1_failures.csv      # Échecs détaillés
results_scenario1_exceptions.csv    # Exceptions
```

### Métriques clés à analyser

| Métrique | Description | SLO |
|----------|-------------|-----|
| **Response Time (p50)** | Médiane temps réponse | < 30s |
| **Response Time (p95)** | 95e percentile | < 60s |
| **Response Time (p99)** | 99e percentile | < 90s |
| **RPS** | Requests Per Second | > 0.5 |
| **Failure Rate** | % échecs | < 5% |
| **Dispatch Duration** | Temps dispatch réel | < 60s |

### Visualisation Web UI

Locust Web UI (http://localhost:8089) affiche :
- **Charts** : Response time, RPS en temps réel
- **Statistics** : Tableau par endpoint
- **Failures** : Liste échecs
- **Exceptions** : Exceptions Python
- **Current ratio** : Distribution des tâches

---

## 🔧 Configuration Avancée

### Variables d'environnement

```bash
# Timeout HTTP
export LOCUST_TIMEOUT=150

# Log level
export LOG_LEVEL=INFO

# Backend host
export BACKEND_HOST=http://localhost:5000
```

### Fichier de configuration (`locust.conf`)

```ini
[locust]
host = http://localhost:5000
users = 10
spawn-rate = 2
run-time = 10m
headless = true
csv = results
loglevel = INFO
```

**Usage :**
```bash
locust -f tests/load_testing/dispatch_load_test.py --config=locust.conf
```

---

## 🐳 Setup Docker pour OSRM Lent (Scénario 3)

### Option A : Modifier Docker Compose

```yaml
# docker-compose.yml
osrm-slow:
  image: osrm/osrm-backend:latest
  volumes:
    - ./data/osrm:/data
  command: >
    sh -c "
      tc qdisc add dev eth0 root netem delay 500ms &&
      osrm-routed --algorithm mld /data/france.osrm
    "
  cap_add:
    - NET_ADMIN
  ports:
    - "5001:5000"
```

### Option B : Toxiproxy (Proxy de latence)

```bash
# Installer toxiproxy
docker run -d --name toxiproxy -p 8474:8474 -p 5001:5001 shopify/toxiproxy

# Ajouter latence 500ms
curl -X POST http://localhost:8474/proxies \
  -d '{
    "name": "osrm_slow",
    "listen": "0.0.0.0:5001",
    "upstream": "osrm:5000"
  }'

curl -X POST http://localhost:8474/proxies/osrm_slow/toxics \
  -d '{
    "name": "latency",
    "type": "latency",
    "attributes": {"latency": 500}
  }'
```

### Option C : Tests sans latence simulée

Le scénario 3 reste utile pour valider :
- Efficacité du cache OSRM
- Fallback Haversine
- Circuit breaker
- Gestion timeouts

---

## 📈 Analyse et Reporting

### Commandes d'analyse

```bash
# Analyser CSV avec pandas
python -c "
import pandas as pd
df = pd.read_csv('results_scenario1_stats.csv')
print(df[['Name', 'Average Response Time', 'Requests/s']].head(10))
"

# Graphiques avec matplotlib
python analyze_results.py --input results_scenario1_stats.csv
```

### Rapport automatique

```bash
# Générer rapport HTML
locust --html=report.html \
  -f tests/load_testing/dispatch_load_test.py \
  --host=http://localhost:5000 \
  --users=10 \
  --spawn-rate=2 \
  --run-time=5m \
  --headless
```

---

## 🛠️ Troubleshooting

### Problème : Login échoue

**Solution :**
```bash
# Vérifier backend running
curl http://localhost:5000/health

# Créer utilisateur test
docker-compose exec api python scripts/create_test_user.py
```

### Problème : Timeouts constants

**Solution :**
```python
# Augmenter timeout dans test
class MyUser(HttpUser):
    network_timeout = 150.0  # 150s
```

### Problème : Erreurs 500

**Solution :**
```bash
# Vérifier logs backend
docker-compose logs -f api | grep ERROR

# Vérifier DB connection
docker-compose exec api python -c "from ext import db; print(db.engine)"
```

### Problème : Pas de résultats CSV

**Solution :**
```bash
# Vérifier permissions écriture
ls -la results_*.csv

# Spécifier chemin absolu
locust ... --csv=/tmp/results
```

---

## 📝 Best Practices

### 1. Warmup progressif

```bash
# Mauvais : spawn-rate=100 (pic brutal)
# Bon : spawn-rate=2 (montée progressive)
locust ... --users=50 --spawn-rate=2
```

### 2. Run time suffisant

```bash
# Trop court : run-time=1m (pas assez de données)
# Bon : run-time=10m (données statistiquement valides)
locust ... --run-time=10m
```

### 3. Monitoring simultané

```bash
# Terminal 1 : Locust
locust -f dispatch_load_test.py --headless ...

# Terminal 2 : Monitoring backend
docker stats atmr-api-1

# Terminal 3 : Logs temps réel
docker-compose logs -f api | grep DISPATCH
```

### 4. Baseline avant/après

```bash
# 1. Baseline AVANT optimisation
locust ... --csv=baseline_before

# 2. Appliquer optimisation

# 3. Baseline APRÈS optimisation
locust ... --csv=baseline_after

# 4. Comparer
python compare_baselines.py baseline_before baseline_after
```

---

## 🔗 Ressources

- [Documentation Locust](https://docs.locust.io/)
- [C2 Load Testing Plan](../../../C2_LOAD_TESTING_DISPATCH_PLAN.md)
- [Audit Technique](../../../AUDIT_TECHNIQUE_COMPLET_2025.md)

---

**Date :** 7 janvier 2025  
**Version :** 1.0.0  
**Auteur :** Équipe ATMR

