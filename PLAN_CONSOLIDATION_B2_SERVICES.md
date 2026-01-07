# 📊 Plan Consolidation B2 - Services Fragmentés

**Date :** 7 janvier 2025  
**Objectif :** Réduire ~150 services → ~50 modules thématiques  
**Référence :** `AUDIT_TECHNIQUE_COMPLET_2025.md` (Section B2, lignes 1402-1421)

---

## 🎯 Objectif Global

**Avant :** ~150 services dispersés (54 fichiers racine + 10 sous-modules)  
**Après :** ~50 modules organisés par domaine métier  
**Bénéfice attendu :** -70% complexité, navigation intuitive

---

## 📋 Analyse Initiale

### Services Existants (54 fichiers + 10 modules)

#### Fichiers Racine (54)

```
backend/services/
├── ab_testing_service.py
├── access_token_service.py
├── ai.py
├── alerting_service.py
├── api_slo.py
├── auto_reassignment_service.py
├── booking_transfer_service.py
├── cache_invalidation.py
├── clamav_service.py
├── csrf_protection.py
├── db_context.py
├── db_session_metrics.py
├── delay_tools.py
├── dispatch_utils.py
├── eta_service.py
├── event_fanout.py
├── event_handlers_registry.py
├── factories.py
├── feature_flags.py
├── file_validation.py
├── geofencing_service.py
├── geolocation_service.py
├── google_places.py
├── holidays_service.py
├── idempotency_service.py
├── invoice_transfer_service.py
├── location_service.py
├── maps.py
├── ml_features.py
├── ml_monitoring_service.py
├── notification_service.py
├── osrm_client.py
├── partner_invoice_pdf_service.py
├── partner_invoice_service.py
├── partnership_service.py
├── partnership_statement_service.py
├── partnership_stats_service.py
├── pdf_service.py
├── planning_service.py
├── proactive_alerts.py
├── prometheus_metrics.py
├── push_service.py
├── qrbill_service.py
├── refresh_token_service.py
├── safety_guards.py
├── secret_rotation_monitor.py
├── socketio_service.py
├── spam_protection.py
├── vacation_service.py
├── version_check.py
├── weather_service.py
├── websocket_healthcheck.py
├── websocket_metrics.py
├── websocket_rate_limiter.py
└── (+ 2 fichiers utilitaires)
```

#### Modules Existants (10)

```
├── agent_dispatch/        (5 fichiers)
├── analytics/             (4 fichiers)
├── event_handlers/        (6 fichiers)
├── interfaces/            (5 fichiers)
├── ml/                    (4 fichiers)
├── pii_masking/           (1 fichier)
├── rl/                    (14 fichiers)
├── sim/                   (2 fichiers)
├── unified_dispatch/      (✅ Déjà refactorisé B1)
└── (1 module mineur)
```

---

## 🗂️ Plan de Consolidation par Domaine

### Domaine 1️⃣ : **Authentication & Security** (10 → 1 module)

**Services à consolider :**

```
services/security/
├── authentication.py      # ← access_token_service.py + refresh_token_service.py
├── csrf.py                # ← csrf_protection.py
├── spam.py                # ← spam_protection.py
├── idempotency.py         # ← idempotency_service.py
├── safety.py              # ← safety_guards.py
├── secret_rotation.py     # ← secret_rotation_monitor.py
└── pii/                   # ← pii_masking/
    └── masking.py
```

**Priorité :** P1 (sécurité critique)

---

### Domaine 2️⃣ : **Notifications** (4 → 1 module)

**Services à consolider :**

```
services/notifications/
├── core.py                # ← notification_service.py
├── push.py                # ← push_service.py
├── alerts.py              # ← alerting_service.py + proactive_alerts.py
└── interfaces.py          # ← interfaces/notification_interface.py
```

**Priorité :** P1 (forte cohésion)

---

### Domaine 3️⃣ : **Geolocation & Routing** (7 → 1 module)

**Services à consolider :**

```
services/geolocation/
├── core.py                # ← geolocation_service.py + location_service.py
├── geofencing.py          # ← geofencing_service.py
├── maps.py                # ← maps.py
├── google_places.py       # ← google_places.py
├── osrm.py                # ← osrm_client.py
└── interfaces.py          # ← interfaces/geocoding_interface.py + routing_interface.py
```

**Priorité :** P2 (APIs externes)

---

### Domaine 4️⃣ : **Partnerships** (5 → 1 module)

**Services à consolider :**

```
services/partnerships/
├── core.py                # ← partnership_service.py
├── invoices.py            # ← partner_invoice_service.py
├── invoices_pdf.py        # ← partner_invoice_pdf_service.py
├── statements.py          # ← partnership_statement_service.py
└── stats.py               # ← partnership_stats_service.py
```

**Priorité :** P2 (domaine cohérent)

---

### Domaine 5️⃣ : **Documents & Files** (5 → 1 module)

**Services à consolider :**

```
services/documents/
├── pdf.py                 # ← pdf_service.py
├── qrbill.py              # ← qrbill_service.py
├── validation.py          # ← file_validation.py
└── security/
    └── clamav.py          # ← clamav_service.py
```

**Priorité :** P2 (peu couplé)

---

### Domaine 6️⃣ : **Booking & Transfers** (3 → 1 module)

**Services à consolider :**

```
services/booking/
├── transfers.py           # ← booking_transfer_service.py
├── invoices.py            # ← invoice_transfer_service.py
└── (intégration avec bounded context bookings/)
```

**Priorité :** P1 (lié au bounded context)

---

### Domaine 7️⃣ : **Machine Learning** (8 → 1 module consolidé)

**Services à consolider :**

```
services/ml/
├── __init__.py
├── features.py            # ← ml_features.py
├── monitoring.py          # ← ml_monitoring_service.py
├── models/
│   ├── demand_prediction.py    # ← ml/demand_prediction.py
│   ├── eta_delay_model.py      # ← ml/eta_delay_model.py
│   ├── model_registry.py       # ← ml/model_registry.py
│   └── training_metadata.py    # ← ml/training_metadata_schema.py
└── rl/                    # ← rl/ (déjà module, à nettoyer)
    ├── agent.py           # Fusion improved_dqn_agent + distributional_dqn
    ├── networks.py        # Fusion improved_q_network + noisy_networks
    ├── buffer.py          # Fusion replay_buffer + n_step_buffer
    ├── env.py             # ← dispatch_env.py
    ├── tuner.py           # ← hyperparameter_tuner.py
    ├── rewards.py         # ← reward_shaping.py
    ├── logger.py          # ← rl_logger.py
    └── suggestions.py     # ← suggestion_generator.py
```

**Priorité :** P1 (complexité élevée, besoin d'organisation)

---

### Domaine 8️⃣ : **Monitoring & Metrics** (7 → 1 module)

**Services à consolider :**

```
services/monitoring/
├── prometheus.py          # ← prometheus_metrics.py
├── db_metrics.py          # ← db_session_metrics.py
├── websocket_metrics.py   # ← websocket_metrics.py
├── slo.py                 # ← api_slo.py
└── websocket/
    ├── healthcheck.py     # ← websocket_healthcheck.py
    └── rate_limiter.py    # ← websocket_rate_limiter.py
```

**Priorité :** P2 (observabilité)

---

### Domaine 9️⃣ : **WebSocket & Real-time** (3 → 1 module)

**Services à consolider :**

```
services/realtime/
├── socketio.py            # ← socketio_service.py
├── healthcheck.py         # ← websocket_healthcheck.py (dupliqué avec monitoring)
└── rate_limiter.py        # ← websocket_rate_limiter.py (dupliqué avec monitoring)
```

**Priorité :** P2 (déjà petit)

---

### Domaine 🔟 : **Analytics** (5 → Déjà module, à valider)

**Services existants :**

```
services/analytics/         # ✅ Déjà module
├── __init__.py
├── aggregator.py
├── insights.py
├── metrics_collector.py
└── report_generator.py
```

**Action :** Valider structure + ajouter documentation

**Priorité :** P3 (déjà organisé)

---

### Domaine 1️⃣1️⃣ : **Dispatch** (6 → 1 module consolidé)

**Services à consolider :**

```
services/dispatch/
├── unified/               # ✅ Déjà refactorisé (B1)
├── agent/                 # ← agent_dispatch/
├── planning.py            # ← planning_service.py
├── auto_reassignment.py   # ← auto_reassignment_service.py
├── utils.py               # ← dispatch_utils.py
└── proactive_alerts.py    # ← proactive_alerts.py (ou → notifications?)
```

**Priorité :** P1 (core business)

---

### Domaine 1️⃣2️⃣ : **Event Handling** (3 → Déjà module, à valider)

**Services existants :**

```
services/events/           # ✅ Déjà module (event_handlers)
├── __init__.py
├── handlers/
│   ├── assignment.py      # ← assignment_handlers.py
│   ├── booking.py         # ← booking_handlers.py
│   ├── dispatch.py        # ← dispatch_handlers.py
│   ├── driver.py          # ← driver_handlers.py
│   └── metrics.py         # ← metrics_handler.py
├── fanout.py              # ← event_fanout.py
└── registry.py            # ← event_handlers_registry.py
```

**Action :** Renommer `event_handlers/` → `events/` + consolider

**Priorité :** P2

---

### Domaine 1️⃣3️⃣ : **External APIs** (3 → 1 module)

**Services à consolider :**

```
services/external/
├── weather.py             # ← weather_service.py + interfaces/weather_interface.py
├── holidays.py            # ← holidays_service.py
└── ai.py                  # ← ai.py (si API externe)
```

**Priorité :** P3 (peu critique)

---

### Domaine 1️⃣4️⃣ : **Utilities & Infrastructure** (8 → 1 module)

**Services à consolider :**

```
services/infrastructure/
├── cache.py               # ← cache_invalidation.py
├── db_context.py          # ← db_context.py
├── feature_flags.py       # ← feature_flags.py
├── factories.py           # ← factories.py
├── version.py             # ← version_check.py
├── ab_testing.py          # ← ab_testing_service.py
└── sim/                   # ← sim/ (simulation)
    └── day_replayer.py
```

**Priorité :** P3 (transverse)

---

### Domaine 1️⃣5️⃣ : **Business Services** (4 → À évaluer)

**Services spécifiques :**

```
├── eta_service.py         # → services/dispatch/eta.py ?
├── delay_tools.py         # → services/dispatch/delay.py ?
├── vacation_service.py    # → services/hr/ (nouveau) ?
```

**Priorité :** P3 (à analyser contexte métier)

---

## 📊 Résumé Consolidation

| Domaine               | Services Avant | Modules Après | Réduction | Priorité |
| --------------------- | -------------- | ------------- | --------- | -------- |
| **Authentication**    | 10             | 1             | -90%      | P1       |
| **Notifications**     | 4              | 1             | -75%      | P1       |
| **Geolocation**       | 7              | 1             | -86%      | P2       |
| **Partnerships**      | 5              | 1             | -80%      | P2       |
| **Documents**         | 5              | 1             | -80%      | P2       |
| **Booking**           | 3              | 1             | -67%      | P1       |
| **Machine Learning**  | 8              | 1             | -88%      | P1       |
| **Monitoring**        | 7              | 1             | -86%      | P2       |
| **WebSocket**         | 3              | 1             | -67%      | P2       |
| **Analytics**         | 5 (module)     | 1 (validé)    | 0%        | P3       |
| **Dispatch**          | 6              | 1             | -83%      | P1       |
| **Events**            | 8 (module)     | 1 (consolidé) | -13%      | P2       |
| **External APIs**     | 3              | 1             | -67%      | P3       |
| **Infrastructure**    | 8              | 1             | -88%      | P3       |
| **Business Services** | 4              | À évaluer     | TBD       | P3       |

**Total :** ~80 services → ~15 modules = **-81% réduction** 🎯

---

## 🚀 Plan d'Exécution (4 semaines)

### Semaine 1 - Domaines P1 (Critiques)

**Jour 1-2 : Authentication & Security**

- Créer `services/security/`
- Migrer 10 services
- Tests unitaires

**Jour 3-4 : Notifications**

- Créer `services/notifications/`
- Migrer 4 services
- Tests unitaires

**Jour 5 : Booking**

- Créer `services/booking/`
- Migrer 3 services
- Tests unitaires

---

### Semaine 2 - Domaines P1 (Business)

**Jour 1-2 : Machine Learning**

- Consolider `services/ml/`
- Migrer 8 services + RL
- Tests unitaires

**Jour 3-5 : Dispatch**

- Consolider `services/dispatch/`
- Migrer 6 services
- Tests intégration

---

### Semaine 3 - Domaines P2 (Support)

**Jour 1 : Geolocation**

- Créer `services/geolocation/`
- Migrer 7 services

**Jour 2 : Partnerships**

- Créer `services/partnerships/`
- Migrer 5 services

**Jour 3 : Documents**

- Créer `services/documents/`
- Migrer 5 services

**Jour 4 : Monitoring**

- Créer `services/monitoring/`
- Migrer 7 services

**Jour 5 : Events**

- Renommer + consolider `events/`
- Migrer 8 services

---

### Semaine 4 - Domaines P3 + Finalisation

**Jour 1 : WebSocket & External APIs**

- Créer modules restants
- Migrer ~6 services

**Jour 2 : Infrastructure**

- Créer `services/infrastructure/`
- Migrer 8 services

**Jour 3-4 : Tests & Documentation**

- Tests intégration complets
- Documentation architecture
- Guide migration

**Jour 5 : Review & Deploy**

- Code review
- Validation équipe
- Merge

---

## 📝 Scripts & Outils

### Script Analyse Dépendances

```bash
# Analyser imports pour identifier couplages
python scripts/analyze-service-dependencies.py
```

### Script Migration

```bash
# Migrer service avec historique Git
./scripts/migrate-service.sh <old_path> <new_path>
```

---

## ✅ Critères de Succès

| Critère                 | Objectif     | Validation       |
| ----------------------- | ------------ | ---------------- |
| **Réduction services**  | -70%         | 80 → ~15 modules |
| **Erreurs compilation** | 0            | Pytest PASS      |
| **Tests unitaires**     | 100% passent | CI/CD            |
| **Documentation**       | 1 doc/module | 15 docs          |
| **Historique Git**      | Préservé     | `git mv`         |

---

## 🎯 Prochaines Étapes Immédiates

1. **Valider plan** avec équipe
2. **Créer scripts** migration
3. **Commencer Semaine 1** (Authentication)

---

**Date de création :** 7 janvier 2025  
**Status :** 🔵 **PLANIFIÉ** - Prêt à démarrer  
**Durée estimée :** 4 semaines progressives
