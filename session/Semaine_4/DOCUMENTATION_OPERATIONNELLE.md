# 📚 DOCUMENTATION OPÉRATIONNELLE - SYSTÈME ML

**Version** : 1.0  
**Date** : 20 Octobre 2025  
**Public** : Équipe technique et opérationnelle

---

## 📋 TABLE DES MATIÈRES

1. [Vue d'ensemble](#vue-densemble)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Utilisation](#utilisation)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)

---

## 🎯 VUE D'ENSEMBLE

### Architecture Système ML

```
┌─────────────────────────────────────────────┐
│              BOOKING REQUEST                │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│          FEATURE FLAGS (Redis)              │
│  ML_ENABLED: true                           │
│  ML_TRAFFIC_PERCENTAGE: 10%-100%            │
└────────────────┬────────────────────────────┘
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
┌──────────┐         ┌──────────┐
│    ML    │         │ FALLBACK │
│  PATH    │         │ (Heurist)│
└────┬─────┘         └─────┬────┘
     │                     │
     ▼                     │
┌─────────────────┐        │
│  WEATHER API    │        │
│ (OpenWeather)   │        │
└────┬────────────┘        │
     │                     │
     ▼                     │
┌─────────────────┐        │
│  ML PREDICTOR   │        │
│  - Features     │        │
│  - Model        │        │
│  - Prediction   │        │
└────┬────────────┘        │
     │                     │
     └──────────┬──────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│           PREDICTION RESULT                 │
│  - delay_minutes                            │
│  - confidence                               │
│  - risk_level                               │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│         LOGGING & MONITORING                │
└─────────────────────────────────────────────┘
```

### Composants Clés

| Composant           | Fichier                                     | Description            |
| ------------------- | ------------------------------------------- | ---------------------- |
| **Feature Flags**   | `feature_flags.py`                          | Contrôle activation ML |
| **Weather Service** | `services/weather_service.py`               | API météo temps réel   |
| **ML Features**     | `services/ml_features.py`                   | Feature engineering    |
| **ML Predictor**    | `services/unified_dispatch/ml_predictor.py` | Prédictions            |
| **A/B Testing**     | `services/ab_testing_service.py`            | Comparaisons           |
| **Monitoring**      | `services/ml_monitoring_service.py`         | Métriques              |

---

## 🔧 INSTALLATION

### 1. Prérequis

```bash
# Docker & Docker Compose
docker --version  # >= 20.10
docker-compose --version  # >= 1.29

# PostgreSQL
psql --version  # >= 13

# Python (dans container)
docker exec atmr-api-1 python --version  # >= 3.11
```

### 2. Clone & Setup

```bash
# Clone repository
git clone <repo_url>
cd atmr

# Build containers
docker-compose build

# Démarrer services
docker-compose up -d
```

### 3. Base de Données

```bash
# Migrations
docker exec atmr-api-1 flask db upgrade

# Vérifier tables ML
docker exec atmr-api-1 psql $DATABASE_URL -c "\dt ml_prediction"
docker exec atmr-api-1 psql $DATABASE_URL -c "\dt ab_test_result"
```

### 4. Dépendances Python

```bash
# Installer scikit-learn (si pas déjà fait)
docker exec atmr-api-1 pip install scikit-learn

# Vérifier installation
docker exec atmr-api-1 python -c "import sklearn; print('sklearn version:', sklearn.__version__)"
```

### 5. Modèle ML

```bash
# Vérifier modèle présent
docker exec atmr-api-1 ls -lh data/ml/models/delay_predictor.pkl

# Si absent, entraîner
docker exec atmr-api-1 python scripts/ml/train_model.py
```

---

## ⚙️ CONFIGURATION

### 1. Variables d'Environnement

**Fichier** : `backend/.env`

```bash
# ===== ML CONFIGURATION =====
ML_ENABLED=true
ML_TRAFFIC_PERCENTAGE=10
FALLBACK_ON_ERROR=true

# ===== OPENWEATHERMAP API =====
OPENWEATHER_API_KEY=your_32_char_key_here

# ===== DATABASE =====
DATABASE_URL=postgresql://user:password@db:5432/atmr

# ===== REDIS =====
REDIS_URL=redis://redis:6379/0

# ===== MONITORING (Optionnel) =====
SENTRY_DSN=your_sentry_dsn
```

### 2. Configuration Feature Flags

**Méthode 1 : Script CLI**

```bash
# Activer ML à 10%
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 10

# Voir statut
docker exec atmr-api-1 python scripts/activate_ml.py --status
```

**Méthode 2 : API**

```bash
# Activer ML
curl -X POST http://localhost:5000/api/feature-flags/ml/enable \
  -H "Content-Type: application/json" \
  -d '{"percentage": 10}'

# Désactiver ML
curl -X POST http://localhost:5000/api/feature-flags/ml/disable
```

### 3. Configuration OpenWeatherMap

**Obtenir API Key** :

1. Créer compte : https://openweathermap.org/
2. Copier clé : https://home.openweathermap.org/api_keys
3. Attendre 10-15 min (activation)

**Configurer** :

```bash
# Script interactif
docker exec -it atmr-api-1 python scripts/setup_weather_api.py

# Ou manuel dans backend/.env
echo "OPENWEATHER_API_KEY=your_key" >> backend/.env

# Redémarrer
docker-compose restart api
```

**Tester** :

```bash
docker exec atmr-api-1 python tests/test_weather_service.py
```

---

## 🚀 UTILISATION

### Activation ML

**Rollout progressif recommandé** :

```bash
# Semaine 1 : 10%
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 10

# Semaine 2 : 25%
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 25

# Semaine 3 : 50%
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 50

# Semaine 4 : 100%
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 100
```

### Vérification Statut

```bash
# CLI
docker exec atmr-api-1 python scripts/activate_ml.py --status

# API
curl http://localhost:5000/api/feature-flags/status | jq

# Logs
docker logs atmr-api-1 | grep "\[FeatureFlag\]"
```

### Tests A/B

```bash
# Exécuter 50 tests
docker exec atmr-api-1 python scripts/ml/run_ab_tests.py --limit 50

# Voir rapport
docker exec atmr-api-1 cat data/ml/ab_test_report.txt
```

---

## 📊 MONITORING

### Dashboard Web

**URL** : http://localhost:3000/ml-monitoring

**Métriques** :

- MAE (dernières 24h)
- R² Score (dernières 24h)
- Temps prédiction moyen
- Taux erreur ML
- Anomalies détectées
- Feature flags status

**Rafraîchissement** : Automatique (30s)

### API Monitoring

```bash
# Résumé complet
curl http://localhost:5000/api/ml-monitoring/summary | jq

# Métriques 24h
curl http://localhost:5000/api/ml-monitoring/metrics?hours=24 | jq

# Anomalies
curl http://localhost:5000/api/ml-monitoring/anomalies?threshold_mae=5.0 | jq

# Prédictions récentes
curl http://localhost:5000/api/ml-monitoring/predictions?limit=50 | jq
```

### Logs

```bash
# Logs ML en temps réel
docker logs -f atmr-api-1 | grep "\[ML\]"

# Logs météo
docker logs -f atmr-api-1 | grep "\[Weather\]"

# Logs feature flags
docker logs -f atmr-api-1 | grep "\[FeatureFlag\]"

# Logs A/B Testing
docker logs -f atmr-api-1 | grep "\[AB Test\]"
```

### Métriques Clés

**À surveiller quotidiennement** :

- ✅ MAE < 2.5 min
- ✅ R² > 0.68
- ✅ Temps prédiction < 1s
- ✅ Taux erreur < 20%
- ✅ Uptime > 99.9%

**À analyser hebdomadairement** :

- Évolution MAE/R² (tendances)
- ROI partiel vs projeté
- Satisfaction client (feedback)
- Anomalies et patterns

---

## 🔧 TROUBLESHOOTING

### Problème 1 : ML ne prédit pas

**Symptôme** : Logs indiquent "Model not trained"

**Solution** :

```bash
# Vérifier modèle présent
docker exec atmr-api-1 ls -lh data/ml/models/delay_predictor.pkl

# Si absent, entraîner
docker exec atmr-api-1 python scripts/ml/train_model.py

# Redémarrer
docker-compose restart api
```

---

### Problème 2 : API météo retourne default (0.5)

**Symptôme** : `Is default: True` dans tests

**Solution** :

```bash
# 1. Vérifier API Key chargée
docker exec atmr-api-1 python -c "import os; print('Key:', 'OK' if os.getenv('OPENWEATHER_API_KEY') else 'MANQUANTE')"

# 2. Si manquante, configurer
docker exec -it atmr-api-1 python scripts/setup_weather_api.py

# 3. Redémarrer container
docker-compose restart api

# 4. Attendre 15 min (activation clé si nouvelle)

# 5. Retester
docker exec atmr-api-1 python tests/test_weather_service.py
```

---

### Problème 3 : Performances lentes (> 2s)

**Symptôme** : Dashboard indique temps prédiction > 2s

**Solutions** :

```bash
# 1. Vérifier cache météo
docker exec atmr-api-1 python -c "from services.weather_service import WeatherService; print(WeatherService.get_cache_stats())"

# 2. Vérifier indices DB
docker exec atmr-api-1 psql $DATABASE_URL -c "SELECT * FROM pg_indexes WHERE tablename IN ('booking', 'assignment', 'ml_prediction');"

# 3. Vérifier ressources container
docker stats atmr-api-1

# 4. Si besoin, augmenter RAM
# Éditer docker-compose.yml → services.api.mem_limit: 2G
```

---

### Problème 4 : Taux erreur ML élevé (> 30%)

**Symptôme** : Dashboard indique beaucoup d'anomalies

**Solution** :

```bash
# 1. Désactiver ML temporairement
docker exec atmr-api-1 python scripts/activate_ml.py --disable

# 2. Analyser erreurs
docker logs atmr-api-1 --tail 200 | grep "\[ML\].*ERROR"

# 3. Vérifier données entrée
docker exec atmr-api-1 python -c "
from models.booking import Booking
b = Booking.query.first()
print('Pickup:', b.pickup_lat, b.pickup_lon)
print('Dropoff:', b.dropoff_lat, b.dropoff_lon)
"

# 4. Si nécessaire, ré-entraîner modèle
docker exec atmr-api-1 python scripts/ml/train_model.py

# 5. Réactiver progressivement (5%)
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 5
```

---

### Problème 5 : Dashboard monitoring inaccessible

**Symptôme** : 404 ou erreur chargement

**Solution** :

```bash
# 1. Vérifier backend API
curl http://localhost:5000/api/ml-monitoring/summary

# 2. Vérifier frontend build
cd frontend
npm run build

# 3. Vérifier routing
# S'assurer que Dashboard.jsx est bien importé dans App.jsx

# 4. Redémarrer frontend
docker-compose restart frontend
```

---

## 🔄 MAINTENANCE

### Quotidienne (5 min)

```bash
# Vérifier dashboard
open http://localhost:3000/ml-monitoring

# Vérifier métriques
curl http://localhost:5000/api/ml-monitoring/summary | jq '.metrics_24h'

# Vérifier logs (erreurs)
docker logs atmr-api-1 --since 24h | grep ERROR | wc -l
```

### Hebdomadaire (30 min)

```bash
# 1. Analyser tendances KPIs
curl http://localhost:5000/api/ml-monitoring/daily?days=7 | jq

# 2. Vérifier ROI partiel
# Calculer gains semaine

# 3. A/B Testing (si trafic < 100%)
docker exec atmr-api-1 python scripts/ml/run_ab_tests.py --limit 50

# 4. Backup base de données
docker exec atmr-postgres-1 pg_dump -U user atmr > backup_$(date +%Y%m%d).sql
```

### Mensuelle (2h)

```bash
# 1. Rapport ROI
# Analyser gains réels vs projetés

# 2. Optimisations
# - Analyser feature importance
# - Identifier features faibles
# - Proposer nouvelles features

# 3. Collecte feedback
# - Drivers : prédictions utiles ?
# - Clients : ETA précis ?
# - Ops : gains opérationnels ?

# 4. Planification ré-entraînement
# Si > 500 nouveaux bookings avec retards réels
```

### Trimestrielle (1 semaine)

```bash
# 1. Ré-entraînement modèle
docker exec atmr-api-1 python scripts/ml/collect_training_data.py
docker exec atmr-api-1 python scripts/ml/train_model.py

# 2. A/B Testing nouveau modèle
# Comparer v1.0 vs v2.0

# 3. Validation ROI réel
# Comparer gains réels vs projetés

# 4. Mise à jour documentation
# Leçons apprises, best practices
```

---

## 📊 MÉTRIQUES & KPIs

### Métriques Techniques

| Métrique             | Cible     | Critique |
| -------------------- | --------- | -------- |
| **MAE**              | < 2.5 min | < 5 min  |
| **R² Score**         | > 0.68    | > 0.50   |
| **Temps prédiction** | < 1s      | < 2s     |
| **Taux erreur**      | < 20%     | < 40%    |
| **Uptime**           | > 99.9%   | > 99%    |

### Métriques Business

| Métrique                | Cible    | Mesure           |
| ----------------------- | -------- | ---------------- |
| **Retards anticipés**   | > 75%    | Dashboard        |
| **Satisfaction client** | ↑ 15%    | Feedback         |
| **Surallocation**       | ↓ 32%    | Temps drivers    |
| **ROI**                 | > 3,000% | Gains financiers |

---

## 🎓 FORMATION

### Pour Développeurs (2h)

**Module 1 : Architecture ML** (45 min)

- Feature engineering pipeline
- ML predictor (RandomForest)
- Intégration dispatch

**Module 2 : Feature Flags** (30 min)

- Activation/désactivation
- Traffic percentage
- Statistiques

**Module 3 : Debugging** (45 min)

- Logs ML, météo, A/B
- Dashboard monitoring
- Procédures rollback

### Pour Ops/DevOps (1h)

**Module 1 : Commandes** (30 min)

- Activation/désactivation ML
- Monitoring (dashboard + API)
- Backup & restore

**Module 2 : Incidents** (30 min)

- Procédures rollback
- Escalation
- Communication

### Pour Business (30 min)

**Module 1 : ROI & KPIs** (15 min)

- ROI 3,310% expliqué
- Gains business mesurables
- Dashboard monitoring

**Module 2 : Communication** (15 min)

- Avantages clients (ETA précis)
- Différenciation concurrentielle
- Feedback collection

---

## 📞 CONTACTS & SUPPORT

### Équipe ML

**ML Lead** : [Nom]  
**Email** : [email]  
**Slack** : @ml-lead

### Équipe Ops

**DevOps Lead** : [Nom]  
**Email** : [email]  
**Slack** : @devops-lead

### Channels Slack

- `#tech-ml` : Questions techniques ML
- `#ops-production` : Incidents production
- `#monitoring-alerts` : Alertes automatiques

### Escalation

**Niveau 1** : Équipe Tech (Slack)  
**Niveau 2** : Lead Dev / Architecte  
**Niveau 3** : CTO

---

## 📚 RESSOURCES

### Documentation

- **Semaine 3** : `session/Semaine_3/RAPPORT_FINAL_SEMAINE_3.md`
- **Semaine 4** : `session/Semaine_4/RAPPORT_FINAL_SEMAINE_4.md` (à créer)
- **ROI** : `session/Semaine_4/ANALYSE_ROI_ML.md`
- **Déploiement** : `session/Semaine_4/GUIDE_DEPLOIEMENT_PRODUCTION.md`

### Code Source

- **ML Predictor** : `backend/services/unified_dispatch/ml_predictor.py`
- **ML Features** : `backend/services/ml_features.py`
- **Weather Service** : `backend/services/weather_service.py`
- **Feature Flags** : `backend/feature_flags.py`

### Tests

- **ML Integration** : `backend/tests/test_ml_integration.py`
- **Weather Service** : `backend/tests/test_weather_service.py`
- **Feature Flags** : `backend/tests/test_feature_flags.py`
- **Monitoring** : `backend/tests/test_ml_monitoring.py`

---

## ✅ CHECKLIST OPÉRATIONNELLE

### Quotidien

- [ ] Vérifier dashboard monitoring
- [ ] Analyser métriques 24h (MAE, R², temps)
- [ ] Scanner logs erreurs
- [ ] Vérifier taux erreur < 20%

### Hebdomadaire

- [ ] Analyser tendances KPIs
- [ ] A/B Testing (si < 100% trafic)
- [ ] Rapport gains hebdomadaires
- [ ] Backup base de données

### Mensuel

- [ ] Rapport ROI détaillé
- [ ] Collecte feedback équipe
- [ ] Planification optimisations
- [ ] Revue documentation

### Trimestriel

- [ ] Ré-entraînement modèle
- [ ] Validation ROI réel
- [ ] A/B Testing nouveau modèle
- [ ] Formation équipe (mise à jour)

---

## 🎯 RÉSUMÉ RAPIDE

### Commandes Essentielles

```bash
# Activer ML (10%)
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 10

# Désactiver ML
docker exec atmr-api-1 python scripts/activate_ml.py --disable

# Dashboard
open http://localhost:3000/ml-monitoring

# Métriques
curl http://localhost:5000/api/ml-monitoring/summary | jq

# Logs
docker logs -f atmr-api-1 | grep "\[ML\]"
```

### KPIs Critiques

✅ MAE < 2.5 min  
✅ R² > 0.68  
✅ Temps < 1s  
✅ Erreurs < 20%  
✅ Uptime > 99.9%

### Contacts Urgence

**Niveau 1** : #tech-ml (Slack)  
**Niveau 2** : Lead Dev  
**Niveau 3** : CTO

---

**Version** : 1.0  
**Dernière mise à jour** : 20 Octobre 2025  
**Prochaine révision** : Janvier 2026
