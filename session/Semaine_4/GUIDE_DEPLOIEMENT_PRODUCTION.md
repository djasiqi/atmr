# 🚀 GUIDE DÉPLOIEMENT PRODUCTION - ML PRÉDICTION RETARDS

**Version** : 1.0  
**Date** : 20 Octobre 2025  
**Auteur** : Équipe ML  
**Statut** : Production-Ready ✅

---

## 📋 PRÉ-REQUIS

### Infrastructure

✅ Docker & Docker Compose installés  
✅ PostgreSQL 13+ configuré  
✅ Redis installé et opérationnel  
✅ Python 3.11+ dans containers  
✅ Node.js 18+ pour frontend

### Données

✅ Modèle ML entraîné (`delay_predictor.pkl` - 35.4 MB)  
✅ Données historiques (5,000+ samples)  
✅ API Key OpenWeatherMap configurée  
✅ Base de données migrée (toutes migrations)

### Tests

✅ Tests unitaires : 100% pass  
✅ Tests intégration : 100% pass  
✅ Tests A/B : 4 validations OK  
✅ Tests end-to-end : Tous passent

---

## 🎯 STRATÉGIE DE DÉPLOIEMENT

### Rollout Progressif (Recommandé)

**Principe** : Activer ML progressivement sur 4 semaines

```
Semaine 1 : 10% trafic  → Validation initiale
Semaine 2 : 25% trafic  → Extension prudente
Semaine 3 : 50% trafic  → Validation à échelle
Semaine 4 : 100% trafic → Déploiement complet
```

**Avantages** :

- ✅ Risque minimal
- ✅ Détection précoce problèmes
- ✅ Ajustements progressifs
- ✅ Rollback facile

---

## 📅 PLANNING DÉPLOIEMENT

### Semaine 1 : Validation Initiale (10% trafic)

**Jour 1-2** : Activation

```bash
# 1. Activer ML à 10%
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 10

# 2. Vérifier statut
curl http://localhost:5000/api/feature-flags/status

# 3. Monitorer logs
docker logs -f atmr-api-1 | grep "\[ML\]"
```

**Jour 3-5** : Monitoring

- 📊 Dashboard monitoring (http://localhost:3000/ml-monitoring)
- 📈 Métriques : MAE, R², temps prédiction
- 🔔 Alertes : taux erreur > 20%

**Jour 6-7** : Analyse

- Comparer ML vs Heuristique
- Valider amélioration -32%
- Collecter feedback drivers/clients

**Critères succès** :

- ✅ Taux erreur < 20%
- ✅ MAE < 2.5 min
- ✅ Temps prédiction < 1s
- ✅ Aucun crash système

---

### Semaine 2 : Extension (25% trafic)

**Jour 1** : Augmentation trafic

```bash
# Passer à 25%
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 25
```

**Jour 2-7** : Monitoring continu

- Valider stabilité à 25%
- Analyser ROI partiel
- Ajuster si nécessaire

**Critères succès** :

- ✅ Performances stables
- ✅ ROI > 200% validé
- ✅ Satisfaction client ↑

---

### Semaine 3 : Validation Échelle (50% trafic)

**Jour 1** : Augmentation trafic

```bash
# Passer à 50%
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 50
```

**Jour 2-7** : Validation à grande échelle

- Tester avec volume élevé
- Valider performances
- Optimiser si besoin

**Critères succès** :

- ✅ Infrastructure stable
- ✅ Temps réponse < 1s
- ✅ Aucune dégradation

---

### Semaine 4 : Déploiement Complet (100% trafic)

**Jour 1** : Activation complète

```bash
# Passer à 100%
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 100
```

**Jour 2-7** : Monitoring production

- Valider ROI complet
- Analyser gains réels
- Planifier optimisations

**Critères succès** :

- ✅ 100% trafic ML
- ✅ ROI 3,310% confirmé
- ✅ Équipe autonome

---

## ⚙️ COMMANDES DÉPLOIEMENT

### Activation ML

```bash
# Activer ML avec X% trafic
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage X

# Exemples
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 10
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 25
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 100
```

### Désactivation ML (Rollback)

```bash
# Désactiver ML complètement
docker exec atmr-api-1 python scripts/activate_ml.py --disable

# Vérifier désactivation
curl http://localhost:5000/api/feature-flags/status
```

### Monitoring

```bash
# Dashboard monitoring
open http://localhost:3000/ml-monitoring

# Logs ML
docker logs -f atmr-api-1 | grep "\[ML\]"

# Statistiques
curl http://localhost:5000/api/ml-monitoring/summary | jq
```

### Tests

```bash
# Tests A/B (50 bookings)
docker exec atmr-api-1 python scripts/ml/run_ab_tests.py --limit 50

# Tests intégration
docker exec atmr-api-1 pytest tests/test_ml_integration.py -v

# Tests météo
docker exec atmr-api-1 pytest tests/test_weather_service.py -v
```

---

## 📊 MONITORING & ALERTES

### KPIs à Surveiller

**Techniques** :

- Temps prédiction ML : < 1s
- Taux erreur ML : < 20%
- MAE (Mean Absolute Error) : < 2.5 min
- R² Score : > 0.68
- Uptime : > 99.9%

**Business** :

- Retards anticipés : > 75%
- Satisfaction client : ↑ 15-20%
- Surallocation : ↓ 32%
- ROI : > 3,000%

### Dashboard Monitoring

**URL** : http://localhost:3000/ml-monitoring

**Métriques affichées** :

- MAE, R², temps prédiction (24h)
- Feature flags status
- Anomalies détectées
- Prédictions récentes (100)

**Auto-refresh** : 30 secondes

### Alertes

**Automatiques** :

- Taux erreur > 20% → Email équipe
- Temps prédiction > 2s → Slack #tech
- API météo down → Fallback activé

**Manuelles** :

- Vérification quotidienne dashboard
- Analyse hebdomadaire KPIs
- Rapport mensuel ROI

---

## 🚨 PROCÉDURES ROLLBACK

### Scénario 1 : Taux Erreur Élevé (> 30%)

```bash
# 1. Désactiver ML immédiatement
docker exec atmr-api-1 python scripts/activate_ml.py --disable

# 2. Vérifier logs
docker logs atmr-api-1 --tail 100 | grep ERROR

# 3. Analyser cause
docker exec atmr-api-1 python scripts/ml/analyze_errors.py

# 4. Corriger et retester
# ...

# 5. Réactiver progressivement (5% → 10%)
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 5
```

### Scénario 2 : Performances Dégradées

```bash
# 1. Réduire trafic ML de 50%
# Exemple : 100% → 50%
docker exec atmr-api-1 python scripts/activate_ml.py --enable --percentage 50

# 2. Analyser performances
curl http://localhost:5000/api/ml-monitoring/metrics

# 3. Optimiser si nécessaire
# - Vérifier cache météo
# - Vérifier indices DB
# - Vérifier RAM/CPU containers

# 4. Ré-augmenter progressivement
```

### Scénario 3 : API Météo Indisponible

**Automatique** : Fallback vers `weather_factor = 0.5`

```python
# Le système continue avec valeur neutre
# Pas d'action requise (déjà implémenté)
```

**Vérification** :

```bash
# Tester fallback
docker exec atmr-api-1 python -c "
from services.weather_service import WeatherService
WeatherService.clear_cache()
w = WeatherService.get_weather(0, 0)
print('Is default:', w.get('is_default'))
# Doit afficher: Is default: True
"
```

---

## 🔧 CONFIGURATION PRODUCTION

### Variables d'Environnement

**Fichier** : `backend/.env`

```bash
# ML Configuration
ML_ENABLED=true
ML_TRAFFIC_PERCENTAGE=10  # Ajuster selon semaine
FALLBACK_ON_ERROR=true

# OpenWeatherMap API
OPENWEATHER_API_KEY=your_api_key_here

# PostgreSQL (production)
DATABASE_URL=postgresql://user:pass@host:5432/atmr_prod

# Redis
REDIS_URL=redis://localhost:6379/0

# Monitoring
SENTRY_DSN=your_sentry_dsn_here  # Optionnel
```

### Feature Flags (Production)

```python
# backend/feature_flags.py

ML_ENABLED = True           # Activer ML
ML_TRAFFIC_PERCENTAGE = 10  # Démarrer à 10%
FALLBACK_ON_ERROR = True    # Toujours actif
```

### API Routes

**Feature Flags** :

- GET `/api/feature-flags/status`
- POST `/api/feature-flags/ml/enable`
- POST `/api/feature-flags/ml/disable`

**Monitoring** :

- GET `/api/ml-monitoring/summary`
- GET `/api/ml-monitoring/metrics`
- GET `/api/ml-monitoring/anomalies`

---

## 📋 CHECKLIST DÉPLOIEMENT

### Avant Déploiement

- [ ] Tous les tests passent (unitaires, intégration, e2e)
- [ ] Modèle ML entraîné et validé (R² > 0.68)
- [ ] API météo configurée et testée
- [ ] Base de données migrée (toutes migrations)
- [ ] Docker containers build et running
- [ ] Dashboard monitoring accessible
- [ ] Feature flags configurés (10% initial)
- [ ] Équipe formée et prête
- [ ] Documentation à jour

### Pendant Déploiement (Semaine 1)

- [ ] ML activé à 10%
- [ ] Dashboard monitoring vérifié quotidiennement
- [ ] Logs analysés (erreurs, performances)
- [ ] KPIs surveillés (MAE, R², temps)
- [ ] Feedback drivers/clients collecté
- [ ] Rapport hebdomadaire créé

### Après Déploiement (Semaine 4)

- [ ] ML activé à 100%
- [ ] ROI validé (> 3,000%)
- [ ] Performances stables
- [ ] Équipe autonome
- [ ] Documentation opérationnelle finalisée
- [ ] Plan d'amélioration continue établi

---

## 👥 FORMATION ÉQUIPE

### Développeurs

**Formation requise** (2h) :

- Architecture ML (predictor, features, weather)
- Feature flags (activation, monitoring)
- Debugging (logs, metrics, errors)
- Maintenance (ré-entraînement, updates)

**Documentation** :

- `README.md` - Vue d'ensemble
- `GUIDE_DEPLOIEMENT_PRODUCTION.md` - Ce guide
- `session/Semaine_3/RAPPORT_FINAL_SEMAINE_3.md` - ML Dev
- `session/Semaine_4/ANALYSE_ROI_ML.md` - ROI

### Ops/DevOps

**Formation requise** (1h) :

- Commandes activation/désactivation ML
- Dashboard monitoring
- Procédures rollback
- Alertes et incidents

### Business/Managers

**Formation requise** (30min) :

- ROI et gains business
- Dashboard monitoring (lecture)
- KPIs à surveiller
- Communication clients

---

## 📞 SUPPORT & CONTACTS

### Équipe Technique

**ML Lead** : [Nom] - [email]  
**DevOps** : [Nom] - [email]  
**Backend** : [Nom] - [email]

### Escalation

**Niveau 1** : Équipe Tech (Slack #tech-ml)  
**Niveau 2** : Lead Dev / Architecte  
**Niveau 3** : CTO

### Ressources

**Documentation** : `session/Semaine_4/`  
**Tests** : `backend/tests/test_ml*.py`  
**Logs** : `docker logs atmr-api-1`  
**Monitoring** : http://localhost:3000/ml-monitoring

---

## 🎯 RÉSUMÉ EXÉCUTIF

### Stratégie

**Rollout progressif 4 semaines** : 10% → 25% → 50% → 100%

### ROI Attendu

**Investissement** : 12,260 CHF  
**Gains annuels** : 418,125 CHF  
**ROI** : **3,310%**  
**Breakeven** : **< 1 semaine**

### Risques & Mitigations

| Risque             | Probabilité | Impact | Mitigation           |
| ------------------ | ----------- | ------ | -------------------- |
| Erreurs ML élevées | Faible      | Élevé  | Rollback automatique |
| API météo down     | Moyen       | Faible | Fallback neutre      |
| Performances       | Faible      | Moyen  | Monitoring continu   |
| Adoption équipe    | Faible      | Moyen  | Formation 2h         |

### Critères Succès

✅ Taux erreur < 20%  
✅ ROI > 3,000% validé  
✅ Satisfaction client ↑ 15%  
✅ Équipe autonome

---

**Version** : 1.0  
**Dernière mise à jour** : 20 Octobre 2025  
**Prochaine révision** : Janvier 2026 (3 mois post-production)
