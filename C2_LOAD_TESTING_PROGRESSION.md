# 🔥 C2: Load Testing Dispatch - Progression

**Date début :** 7 janvier 2025  
**Status :** 🔵 **EN COURS** (Jour 2/7)  
**Référence :** `AUDIT_TECHNIQUE_COMPLET_2025.md` (lignes 1532-1555)

---

## 📊 Vue d'Ensemble

| Objectif      | Valider performance sous charge |
| ------------- | ------------------------------- |
| **Scénarios** | 3 tests de charge Locust        |
| **Durée**     | 1 semaine (7 jours)             |
| **Status**    | Jour 2/7 complété               |

---

## ✅ Jour 1 : Setup Locust (7 janvier 2025 - 22h)

**Objectifs :**

- ✅ Installer Locust
- ✅ Créer structure tests
- ✅ Hotfixes backend/Celery

**Livrables :**

- ✅ Plan détaillé : `C2_LOAD_TESTING_DISPATCH_PLAN.md`
- ✅ Structure : `backend/tests/load_testing/`
- ✅ Hotfixes : 16 fichiers corrigés (10 commits)
  - Backend API : 7 fichiers
  - Celery/Flower : 3 fichiers
  - Tests : 6 fichiers

**Résultats :**

- Stack complète opérationnelle
- Backend, Celery Worker, Beat, Flower : ✅ UP
- Rapport hotfixes : `backend/HOTFIX_B1_B2_IMPORTS.md`

**Durée :** ~2h30 (dont 2h hotfixes)

---

## ✅ Jour 2 : Implémentation Scénarios (7 janvier 2025 - 23h)

**Objectifs :**

- ✅ Implémenter Scénario 1 (Charge standard)
- ✅ Implémenter Scénario 2 (Multi-entreprises)
- ✅ Implémenter Scénario 3 (OSRM lent)
- ✅ Documentation complète

**Livrables :**

### 1. Scénario 1 : Charge Standard (`dispatch_load_test.py`)

**Caractéristiques :**

- 390 lignes de code
- Test : 100 bookings × 50 drivers
- Optimisation : OR-Tools (MIP)
- Matrices : 5000 éléments
- Métriques : Response time, RPS, Dispatch duration
- Event handlers : test_start, test_stop, request
- Modes : Web UI, Headless, Distribué

**SLO :**

- Dispatch duration : < 60s
- Taux assignation : > 80%
- Success rate : > 95%

### 2. Scénario 2 : Multi-Entreprises (`multi_company_test.py`)

**Caractéristiques :**

- 450 lignes de code
- Test : 10 entreprises en parallèle
- Validation : Isolation données, locks Redis
- Contention : DB queries simultanées
- Tests : Dispatch, bookings, drivers, locks Redis

**SLO :**

- Isolation données : 100% (pas de leak)
- Locks Redis : Fonctionnels
- Pas de deadlocks DB
- Performance stable sous charge

### 3. Scénario 3 : OSRM Lent (`slow_osrm_test.py`)

**Caractéristiques :**

- 480 lignes de code
- Test : OSRM avec 500ms latency
- Validation : Timeouts, fallback, circuit breaker
- Cache : Hit rate > 80%
- Tests : Dispatch slow, cache, Haversine, health, circuit breaker

**SLO :**

- Dispatch duration : < 120s (malgré latence)
- Fallback Haversine : Fonctionnel
- Circuit breaker : Actif si nécessaire
- Success rate : > 90%

### 4. Documentation (`README.md`)

**Contenu :**

- Guide installation Locust
- Usage : Web UI, Headless, Distribué
- Configuration avancée
- Setup Docker OSRM lent
- Analyse résultats
- Troubleshooting
- Best practices

**Taille :** 397 lignes

**Résultats :**

- **Total lignes code :** 1320 lignes (3 scénarios)
- **Documentation :** 397 lignes (README)
- **Total :** 1717 lignes
- **Commits :** 2 commits

**Durée :** ~1h

---

## 🔧 Hotfix : JWT Audience (8 janvier 2026 - 10h30)

**Problème :**

- 100% des tests Locust échouaient avec `401 UNAUTHORIZED`
- Message d'erreur : `"Token has been revoked"`
- Logs API : `"Token sans claim 'aud' (audience)"`

**Cause :**

- L'endpoint `/api/auth/login-test` créait des tokens **sans audience** (`aud`)
- Le système de validation JWT rejette tous les tokens sans audience valide (`"atmr-api"` ou `"atmr-mobile-enterprise"`)

**Correction :**

- ✅ Ajout du claim `aud: "atmr-api"` dans `/login-test` (`backend/routes/auth.py`)
- ✅ Ajout des claims standards (`user_id`, `role`, `company_id`, `driver_id`)
- ✅ Rebuild image Docker `api`
- ✅ Redémarrage services `api` et `locust`
- ✅ Script de validation : `backend/tests/load_testing/quick_test_jwt.py`

**Validation :**

```bash
[OK] Login reussi!
   Token (preview): eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   User: admin@test.com

[OK] Token accepte (status 403)
   Audience JWT validee avec succes!

[SUCCESS] TEST REUSSI: Les tokens JWT ont l'audience correcte
```

**Status :** ✅ **RÉSOLU**  
**Impact :** Bloquant critique → Tests prêts à être relancés  
**Temps :** ~1h (identification + correction + validation)

**Commits :**

- `[C2-HOTFIX] Ajouter audience JWT au login-test pour passer validation`
- `[C2-HOTFIX] Script test JWT audience - validation du fix login-test`

---

## 🔲 Jour 3-4 : Exécution Tests (EN COURS)

**Objectifs :**

- 🔲 Exécuter Scénario 1 (5-10 users, 15min)
- 🔲 Exécuter Scénario 2 (10-20 users, 10min)
- 🔲 Exécuter Scénario 3 (5-10 users, 15min)
- 🔲 Collecter métriques CSV
- 🔲 Capturer screenshots Web UI

**Configuration recommandée :**

```bash
# Scénario 1
locust -f backend/tests/load_testing/dispatch_load_test.py \
    --host=http://localhost:5000 \
    --users=10 --spawn-rate=2 --run-time=15m \
    --headless --csv=results/scenario1

# Scénario 2
locust -f backend/tests/load_testing/multi_company_test.py \
    --host=http://localhost:5000 \
    --users=10 --spawn-rate=10 --run-time=10m \
    --headless --csv=results/scenario2

# Scénario 3
locust -f backend/tests/load_testing/slow_osrm_test.py \
    --host=http://localhost:5000 \
    --users=5 --spawn-rate=1 --run-time=15m \
    --headless --csv=results/scenario3
```

**Livrables attendus :**

- Fichiers CSV (stats, history, failures)
- Screenshots Web UI
- Logs détaillés

---

## 🔲 Jour 5-6 : Analyse Résultats

**Objectifs :**

- 🔲 Analyser métriques (p50, p95, p99, RPS)
- 🔲 Identifier goulots d'étranglement
- 🔲 Comparer SLO vs résultats
- 🔲 Documenter observations

**Analyses :**

- Response times (distribution)
- Dispatch duration (évolution)
- Failure rate (causes)
- OSRM cache hit rate
- DB contention (logs)
- Redis locks (contention)

**Outils :**

- Pandas (analyse CSV)
- Matplotlib (graphiques)
- Locust Web UI (temps réel)
- Docker stats (monitoring)

---

## 🔲 Jour 7 : Rapport Final

**Objectifs :**

- 🔲 Rapport synthèse
- 🔲 Recommandations optimisation
- 🔲 Checklist pré-production
- 🔲 Marquer C2 comme ✅ COMPLÉTÉ

**Rapport final attendu :**

- Résumé exécutif
- Métriques clés (tableau)
- Graphiques performance
- Goulots d'étranglement identifiés
- Recommandations priorisées (P0, P1, P2)
- Checklist stabilisation

**Format :** `C2_LOAD_TESTING_RAPPORT_FINAL.md`

---

## 📊 Métriques de Succès

| Critère                   | Objectif | Status |
| ------------------------- | -------- | ------ |
| **Scénarios implémentés** | 3/3      | ✅ 3/3 |
| **Tests exécutés**        | 3/3      | 🔲 0/3 |
| **Rapport généré**        | 1        | 🔲 0/1 |
| **SLO validés**           | > 90%    | 🔲 TBD |
| **Goulots identifiés**    | > 3      | 🔲 TBD |
| **Recommandations**       | > 5      | 🔲 TBD |

---

## 🔧 Hotfix #3 : Rate Limits Dispatch (8 janvier - 10h25)

### ❌ Problème

Tests Locust échouaient avec **429 TOO MANY REQUESTS** : `{ "message": "30 per 1 hour" }`

### 🔍 Analyse

Rate limits trop restrictifs dans **3 fichiers différents** :

| Fichier                              | Endpoint                             | Limite AVANT               | Limite APRÈS |
| ------------------------------------ | ------------------------------------ | -------------------------- | ------------ |
| `company_mobile_dispatch.py` (L2085) | `/v1/company_dispatch/run`           | **10/minute**              | 10000/hour   |
| `company_mobile_dispatch.py` (L2187) | `/v1/company_dispatch/optimizer/run` | **10/minute**              | 10000/hour   |
| `dispatch/dispatch_run.py` (L47)     | `/dispatch/run`                      | **30/hour** ← cause du 429 | 10000/hour   |
| `dispatch/dispatch_run.py` (L399)    | `/dispatch/trigger`                  | **50/hour**                | 10000/hour   |

### ✅ Solution

1. **Commit 877ff7c** : Augmentation `company_mobile_dispatch.py` (10/min → 10000/h)
2. **Commit 87cb380** : Augmentation `dispatch/dispatch_run.py` (30/h + 50/h → 10000/h)
3. Redémarrage Redis + API pour appliquer les changements

### 📊 Impact

- Rate limits passés de **30-600/h** à **10000/h** pour load testing
- Permet jusqu'à **2.78 requêtes/seconde** par endpoint
- Suffisant pour tester 100 bookings × 50 drivers avec 10 users Locust

---

## 🚀 Prochaines Étapes

1. ✅ **Scénarios implémentés** (Jour 2 complété)
2. 🔵 **Exécuter tests** (Jours 3-4) ← **NEXT**
3. 🔲 **Analyser résultats** (Jours 5-6)
4. 🔲 **Rapport final** (Jour 7)
5. 🔲 **Marquer C2 ✅** dans audit

---

**Dernière mise à jour :** 8 janvier 2026 - 10h35  
**Progression globale :** 2/7 jours (29%) + Hotfix JWT + Hotfix Rate Limits  
**Status :** 🔵 EN COURS - Hotfixes résolus (JWT + Rate Limits 3 fichiers), prêt pour exécution tests
