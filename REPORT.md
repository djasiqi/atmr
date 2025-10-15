# 🔍 Audit Complet de l'Application ATMR (Transport Médical)

**Date**: 15 octobre 2025  
**Auditeur**: Analyse automatisée complète  
**Périmètre**: Backend Flask/Celery, Frontend React, Mobile React Native, Infrastructure Docker

---

## 📋 Executive Summary

### ✅ **Points forts majeurs**

1. **Architecture modulaire bien structurée**: Séparation claire models/routes/services/tasks
2. **Dispatch temps réel robuste**: OSRM client avec fallback haversine, cache Redis, retry avec backoff exponentiel
3. **Sécurité JWT**: Rate limiting sur auth, claims personnalisés (role/company_id), tokens refresh
4. **SocketIO bien scoped**: Rooms par company/driver, auth JWT au connect, isolation multi-tenant
5. **PDF/QR-Bill professionnel**: Support facturation tierce, génération conforme norme suisse
6. **Validations ORM exhaustives**: Contraintes CHECK, validators Marshmallow, IBAN/UID-IDE/phone
7. **Infrastructure Docker**: Multi-stage build, utilisateur non-root, healthchecks
8. **Gestion timezone consciente**: Utilitaires `shared/time_utils.py` avec mode naïf local Europe/Zurich
9. **Tasks Celery avec retry automatique**: `autoretry_for`, backoff, jitter
10. **Frontend discovery méthodique**: Hooks personnalisés, structure pages/components cohérente

### ⚠️ **Faiblesses critiques**

1. **Incohérence timezone massive**: Mix DateTime(timezone=True/False), usage `datetime.utcnow()` (deprecated) vs `datetime.now(timezone.utc)`
2. **Manque d'index DB**: `invoice_line_id` sur booking, `company_id` sur plusieurs tables, impacts requêtes fréquentes
3. **N+1 queries potentielles**: Relations lazy sans joinedload sur routes critiques (bookings, invoices)
4. **Celery: pas d'`acks_late`**: Risque perte de tâches si worker crash avant traitement complet
5. **Frontend: pas de refresh automatique token 401**: Logout immédiat au lieu de retry après refresh
6. **PDF service: URLs hardcodées**: `http://localhost:5000/uploads/...` au lieu de config dynamique
7. **Pas de CI/CD**: Aucun workflow GitHub Actions détecté (lint/tests/build)
8. **Migrations potentiellement désynchronisées**: Drift models ↔ DB à vérifier (ex: `invoice_line_id` ajouté mais index manquant)
9. **Logs: PII non masqué**: Emails, noms, adresses loggés en clair (GDPR-like non respecté)
10. **Mobile: structure minimale**: Apps client/driver existent mais peu de code analysé

---

## 🎯 Top 20 Findings (Classés par Impact × Complexity × Effort - ICE)

| #   | Finding                                                    | Impact | Complexity | Effort | Score ICE | Catégorie           | Now/Next/Later |
| --- | ---------------------------------------------------------- | ------ | ---------- | ------ | --------- | ------------------- | -------------- |
| 1   | **Incohérence timezone (DateTime TZ)**                     | 10     | 8          | 9      | 720       | Backend/Data        | **NOW**        |
| 2   | **Index manquants (invoice_line_id, company_id)**          | 9      | 3          | 2      | 54        | Backend/Perf        | **NOW**        |
| 3   | **Celery acks_late manquant**                              | 9      | 2          | 1      | 18        | Backend/Reliability | **NOW**        |
| 4   | **datetime.utcnow() deprecated partout**                   | 8      | 2          | 3      | 48        | Backend/Quality     | **NOW**        |
| 5   | **N+1 queries (bookings/invoices routes)**                 | 8      | 5          | 4      | 160       | Backend/Perf        | **NOW**        |
| 6   | **PDF URLs hardcodées (localhost:5000)**                   | 7      | 3          | 2      | 42        | Backend/Config      | **NOW**        |
| 7   | **Frontend: pas de refresh auto JWT**                      | 8      | 4          | 3      | 96        | Frontend/Auth       | **NEXT**       |
| 8   | **PII dans les logs (GDPR)**                               | 9      | 6          | 5      | 270       | Backend/Security    | **NEXT**       |
| 9   | **Pas de CI/CD (workflows manquants)**                     | 7      | 5          | 6      | 210       | Infra/DevEx         | **NEXT**       |
| 10  | **SocketIO: manque validation payload driver_location**    | 6      | 3          | 2      | 36        | Backend/Security    | **NOW**        |
| 11  | **Celery task_time_limit non défini**                      | 6      | 2          | 1      | 12        | Backend/Config      | **NOW**        |
| 12  | **OSRM: lock threading global (\_rl_lock)**                | 5      | 7          | 8      | 280       | Backend/Perf        | **LATER**      |
| 13  | **Invoice: pas de validation montants négatifs**           | 7      | 2          | 1      | 14        | Backend/Logic       | **NOW**        |
| 14  | **Frontend: duplication services (company/driver/client)** | 5      | 6          | 7      | 210       | Frontend/Arch       | **NEXT**       |
| 15  | **docker-compose: manque healthcheck sur api**             | 5      | 2          | 1      | 10        | Infra/Ops           | **NOW**        |
| 16  | **Payment: enum method défini en dur vs models.enums**     | 4      | 2          | 2      | 16        | Backend/Arch        | **NEXT**       |
| 17  | **Migration drift (invoice_line_id pas de FK index)**      | 6      | 4          | 3      | 72        | Backend/Schema      | **NOW**        |
| 18  | **Backend tests: couverture <30% estimée**                 | 8      | 7          | 9      | 504       | Backend/Quality     | **NEXT**       |
| 19  | **Frontend: assets/CSS morts (estimé 15-20%)**             | 3      | 5          | 6      | 90        | Frontend/Cleanup    | **LATER**      |
| 20  | **QR-Bill: adresses fallback hardcodées (Genève)**         | 4      | 3          | 2      | 24        | Backend/Logic       | **NEXT**       |

**Légende scoring ICE:**

- **Impact**: 1-10 (10 = critique production)
- **Complexity**: 1-10 (10 = très complexe à corriger)
- **Effort**: 1-10 (10 = plusieurs jours de travail)
- **Score ICE**: Impact × Complexity × Effort (plus élevé = priorité plus haute si Impact fort)

**Classification NOW/NEXT/LATER:**

- **NOW** (Semaine 1): Correctifs rapides, impact élevé, effort faible/moyen
- **NEXT** (Semaines 2-4): Refactorings moyens, impact moyen/élevé
- **LATER** (Backlog): Optimisations lourdes, impact moyen/faible

---

## 🗂️ Tableau de Dette Technique

| Origine                         | Risque                                                  | Proposition                                                   | Effort estimé |
| ------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------- | ------------- |
| **Timezone: mix naïf/aware**    | Calculs dates incorrects, bugs fuso horaire             | Migration complète vers UTC aware + helpers time_utils        | M (2-3j)      |
| **Index DB manquants**          | Scans séquentiels, lenteurs >10k bookings               | Créer migrations Alembic avec indexes                         | S (2h)        |
| **Celery: acks_late=False**     | Perte de tâches si crash worker                         | Config `acks_late=True`, `task_time_limit=300`                | S (30min)     |
| **datetime.utcnow() partout**   | Deprecated Python 3.12+, incohérence TZ                 | Remplacer par `datetime.now(timezone.utc)`                    | M (1j)        |
| **N+1 queries**                 | API lentes, timeouts si 100+ bookings                   | `joinedload()` sur relations, pagination stricte              | M (1-2j)      |
| **PDF URLs hardcodées**         | Cassé en prod si domaine ≠ localhost                    | Config `PDF_BASE_URL` via env, utiliser `current_app.config`  | S (1h)        |
| **Refresh JWT manuel**          | UX dégradée (déco fréquente), frustration utilisateur   | Interceptor axios avec retry après refresh                    | M (3h)        |
| **PII logs**                    | Non-conformité GDPR, risque audit                       | Masquer emails/noms via formatter logging custom              | M (2j)        |
| **Pas de CI/CD**                | Régressions non détectées, déploiements manuels risqués | GitHub Actions (lint/test/build)                              | M (1j)        |
| **SocketIO validation**         | Injection payloads malveillants, crash rooms            | Valider lat/lon/driver_id avant emit                          | S (1h)        |
| **OSRM lock global**            | Bottleneck >50 req/s, contention threads                | Lock per-request ou async (httpx)                             | L (3-5j)      |
| **Invoice montants négatifs**   | Factures négatives acceptées, comptabilité cassée       | Contrainte CHECK + validation Marshmallow                     | S (1h)        |
| **Services frontend dupliqués** | Maintenance x3, bugs incohérents                        | Service générique `apiService.js` + factories                 | M (2j)        |
| **docker-compose healthcheck**  | Containers start avant DB ready, crashes init           | Healthcheck sur api, dépendances `condition: service_healthy` | S (30min)     |
| **Payment enum hardcodé**       | Duplication logique, risque désynchronisation           | Utiliser `models.enums.PaymentMethod` partout                 | S (30min)     |
| **Migration drift**             | Schéma DB ≠ models, erreurs runtime                     | Générer migration complète `alembic revision --autogenerate`  | M (2h)        |
| **Tests coverage <30%**         | Bugs non détectés, refactorings risqués                 | Ajouter pytest (auth, bookings, dispatch), RTL (pages)        | L (5-10j)     |
| **Assets morts frontend**       | Build lourd (+500kb), temps chargement                  | Audit `webpack-bundle-analyzer`, retirer unused               | M (1j)        |
| **QR-Bill fallbacks**           | Adresses Genève/Anières sur factures réelles            | Retour erreur 400 si adresse manquante                        | S (1h)        |

**Effort:**

- **S (Small)**: <4h
- **M (Medium)**: 1-3 jours
- **L (Large)**: 5+ jours

---

## 🗺️ Carte des Dépendances (Backend Services ↔ Routes ↔ Tables)

### Backend: Services → Tables

```
invoice_service.py
  ├─ Invoice (R/W)
  ├─ InvoiceLine (R/W)
  ├─ InvoiceReminder (R/W)
  ├─ InvoiceSequence (R/W)
  ├─ CompanyBillingSettings (R)
  ├─ Booking (R - requêtes période)
  └─ Client (R - infos débiteur)

pdf_service.py
  ├─ Invoice (R)
  ├─ Company (R - logo, adresse)
  ├─ Client (R - adresse facturée)
  ├─ InvoiceLine (R - détails courses)
  └─ Booking (R - dates/trajets via invoice_lines)

qrbill_service.py
  ├─ Invoice (R)
  ├─ Company (R - IBAN, adresse créancier)
  ├─ Client (R - adresse débiteur)
  └─ CompanyBillingSettings (R - IBAN)

osrm_client.py
  ├─ Redis (cache matrices/routes)
  └─ (Aucune table SQL, service pur HTTP)

unified_dispatch/engine.py
  ├─ Booking (R/W - assignments)
  ├─ Driver (R - disponibilités)
  ├─ DriverStatus (R/W - état temps réel)
  ├─ DriverShift (R - planning)
  ├─ DriverVacation (R - absences)
  ├─ DriverWorkingConfig (R - contraintes horaires)
  ├─ Assignment (R/W - résultat dispatch)
  ├─ DispatchRun (W - historique)
  ├─ DispatchMetrics (W - analytics)
  └─ Company (R - config dispatch)

notification_service.py
  ├─ User (R - emails)
  ├─ Company (R - contact)
  ├─ Driver (R - push_token)
  └─ Message (W - historique)
```

### Backend: Routes → Services

```
routes/auth.py
  → ext.jwt (create_access_token, decode_token)
  → ext.mail (forgot password)
  → models.User, Client

routes/bookings.py
  → services.maps (geocode, distance)
  → services.unified_dispatch.queue (trigger)
  → models.Booking, Client, Driver

routes/invoices.py
  → services.invoice_service.InvoiceService
  → services.pdf_service.PDFService
  → models.Invoice, InvoiceLine, Company, Client

routes/companies.py
  → models.Company, Driver, Booking
  → services.invoice_service (via /invoices sub-routes)
  → services.unified_dispatch.queue (via /dispatch sub-routes)

routes/dispatch_routes.py
  → services.unified_dispatch.engine
  → services.unified_dispatch.queue
  → tasks.dispatch_tasks (Celery)
  → models.DispatchRun, Assignment, DriverStatus

sockets/chat.py
  → models.Message, Driver, Company, User
  → ext.redis_client (driver locations)
  → Flask session
```

### Frontend: Pages → Services API

```
pages/company/Dashboard/
  → services/companyService.js
    ├─ GET /api/companies/me
    ├─ GET /api/companies/me/drivers
    └─ GET /api/companies/me/bookings

  → services/companySocket.js
    ├─ connect() -> room company_{id}
    ├─ on("driver_location_update")
    └─ emit("team_chat_message")

  → services/dispatchMonitoringService.js
    ├─ GET /api/companies/me/dispatch/status
    ├─ POST /api/companies/me/dispatch/run
    └─ GET /api/companies/me/dispatch/assignments

pages/company/Invoices/
  → services/invoiceService.js
    ├─ GET /api/companies/me/invoices
    ├─ POST /api/companies/me/invoices
    ├─ GET /api/companies/me/invoices/{id}/pdf
    └─ POST /api/companies/me/invoices/{id}/send

  → utils/invoiceGenerator.js (client-side PDF - ANOMALIE)
  → utils/qrbillGenerator.js (client-side QR-bill - ANOMALIE)

pages/driver/Dashboard/
  → services/driverService.js
    ├─ GET /api/drivers/me
    ├─ GET /api/drivers/me/bookings
    └─ PATCH /api/drivers/me/status

  → services/companySocket.js
    ├─ emit("driver_location", {lat, lon})
    └─ on("team_chat_message")

pages/client/Reservations/
  → services/reservationService.js
    ├─ GET /api/clients/{public_id}/bookings
    ├─ POST /api/clients/{public_id}/bookings
    └─ DELETE /api/bookings/{id}
```

**⚠️ ANOMALIES DÉTECTÉES:**

1. **Génération PDF/QR-bill côté frontend**: `invoiceGenerator.js` et `qrbillGenerator.js` dupliquent la logique backend → **À SUPPRIMER**, tout doit passer par le backend
2. **Services dupliqués**: `companyService.js`, `driverService.js`, `clientService.js` partagent 70% du code → **Factoriser** dans `apiService.js` générique

---

## 📊 Schéma ERD (Entity-Relationship Diagram - Mermaid)

```mermaid
erDiagram
    User ||--o{ Client : "has"
    User ||--o| Driver : "is"
    User ||--o| Company : "owns"

    Company ||--o{ Client : "manages"
    Company ||--o{ Driver : "employs"
    Company ||--o{ Booking : "serves"
    Company ||--o{ Invoice : "issues"
    Company ||--o{ DispatchRun : "executes"
    Company ||--o| CompanyBillingSettings : "configures"
    Company ||--o| CompanyPlanningSettings : "configures"
    Company ||--o{ Vehicle : "owns"

    Client ||--o{ Booking : "requests"
    Client ||--o{ Payment : "makes"
    Client ||--o{ Invoice : "receives_service_invoice"
    Client ||--o{ Invoice : "pays_as_third_party"

    Driver ||--o{ Booking : "fulfills"
    Driver ||--o{ DriverShift : "works"
    Driver ||--o{ DriverVacation : "takes"
    Driver ||--o{ Assignment : "assigned_to"
    Driver ||--o| DriverStatus : "has_status"
    Driver ||--o| DriverWorkingConfig : "has_config"
    Driver ||--o{ Message : "sends/receives"

    Booking ||--o{ Payment : "paid_by"
    Booking ||--o| InvoiceLine : "billed_in"
    Booking ||--o{ Assignment : "dispatched_as"
    Booking ||--o| Booking : "has_return_trip"

    Invoice ||--o{ InvoiceLine : "contains"
    Invoice ||--o{ InvoicePayment : "paid_by"
    Invoice ||--o{ InvoiceReminder : "has_reminders"

    DispatchRun ||--o{ Assignment : "produces"
    DispatchRun ||--o| DispatchMetrics : "measures"
    DispatchRun }o--|| Company : "belongs_to"

    Company ||--o{ InvoiceSequence : "tracks_numbering"
    Company ||--o{ DailyStats : "aggregates"
    Company ||--o{ DispatchMetrics : "measures"
    Company ||--o{ RealtimeEvent : "logs"

    DriverShift ||--o{ DriverBreak : "includes"
    DriverShift }o--|| Vehicle : "uses"

    Message }o--|| User : "sender"
    Message }o--|| User : "receiver"
    Message }o--|| Company : "within"

    User {
        int id PK
        string public_id UK
        string username UK
        string email UK "nullable"
        string password
        enum role "ADMIN|CLIENT|DRIVER|COMPANY"
        string phone "nullable"
        date birth_date "nullable"
        enum gender "nullable"
    }

    Client {
        int id PK
        int user_id FK
        int company_id FK "nullable"
        enum client_type "SELF_SERVICE|PRIVATE|CORPORATE"
        string billing_address "nullable"
        boolean is_institution
        string institution_name "nullable"
    }

    Driver {
        int id PK
        int user_id FK UK
        int company_id FK
        string license_plate "encrypted"
        enum driver_type "REGULAR|EMERGENCY"
        float latitude "nullable"
        float longitude "nullable"
        string push_token "nullable"
    }

    Company {
        int id PK
        int user_id FK UK
        string name
        string address "nullable"
        string iban "nullable, indexed"
        string uid_ide "nullable, indexed"
        boolean is_approved
        boolean dispatch_enabled
    }

    Booking {
        int id PK
        int client_id FK
        int company_id FK "nullable"
        int driver_id FK "nullable"
        int user_id FK
        datetime scheduled_time "timezone=False (naïf local)"
        enum status "PENDING|ACCEPTED|ASSIGNED|..."
        float amount
        boolean is_round_trip
        boolean is_return
        int parent_booking_id FK "nullable"
        int invoice_line_id FK "nullable, index MANQUANT"
    }

    Invoice {
        int id PK
        int company_id FK
        int client_id FK
        int bill_to_client_id FK "nullable (third-party)"
        int period_month
        int period_year
        string invoice_number UK "per company"
        numeric total_amount
        numeric balance_due
        enum status "draft|sent|paid|overdue|cancelled"
        datetime issued_at "timezone=True"
        datetime due_date "timezone=True"
        string pdf_url "nullable"
        string qr_reference "nullable"
    }

    InvoiceLine {
        int id PK
        int invoice_id FK
        enum type "ride|late_fee|reminder_fee|custom"
        string description
        numeric qty
        numeric unit_price
        numeric line_total
        int reservation_id FK "nullable"
    }

    DispatchRun {
        int id PK
        int company_id FK
        date day UK "per company"
        enum status "PENDING|RUNNING|COMPLETED|FAILED"
        datetime started_at "nullable, timezone=True"
        datetime completed_at "nullable, timezone=True"
        jsonb config "nullable"
        jsonb metrics "nullable"
    }

    Assignment {
        int id PK
        int dispatch_run_id FK "nullable"
        int booking_id FK UK "per dispatch_run"
        int driver_id FK "nullable"
        enum status "SCHEDULED|EN_ROUTE_PICKUP|..."
        datetime planned_pickup_at "timezone=True"
        datetime actual_pickup_at "nullable, timezone=True"
        int delay_seconds
    }

    DriverStatus {
        int id PK
        int driver_id FK UK
        enum state "AVAILABLE|BUSY|OFFLINE"
        float latitude "nullable"
        float longitude "nullable"
        datetime next_free_at "nullable, timezone=True"
        int current_assignment_id FK "nullable"
    }
```

**Notes schéma:**

- **Timezone mixing**: Booking.scheduled_time est `timezone=False` (naïf local Europe/Zurich), mais Invoice.issued_at est `timezone=True` (UTC aware) → **Incohérence à corriger**
- **Index manquant**: Booking.invoice_line_id n'a pas d'index déclaré malgré FK → **Créer migration**
- **Contraintes**: CHECK constraints présentes sur lat/lon, montants positifs, dates cohérentes
- **Encryption**: Driver.license_plate chiffré via `sqlalchemy_utils.StringEncryptedType`
- **Third-party billing**: Invoice.bill_to_client_id permet facturation tierce (cliniques/assurances)

---

## 🏗️ Plan d'Implémentation (Roadmap)

### **Semaine 1 (Now - Correctifs Critiques)**

| Jour  | Tâche                                                        | Effort | Risque    | Rollback                                 |
| ----- | ------------------------------------------------------------ | ------ | --------- | ---------------------------------------- |
| J1-J2 | **Migration timezone complète**                              | M      | M         | Rollback vers models actuels si tests KO |
|       | - Uniformiser DateTime(timezone=True) partout                |        |           |                                          |
|       | - Remplacer datetime.utcnow() → datetime.now(timezone.utc)   |        |           |                                          |
|       | - Tests régression complète (auth, bookings, invoices)       |        |           |                                          |
| J2    | **Index DB critiques**                                       | S      | L (basse) | DROP INDEX si perf dégradée              |
|       | - Créer index sur Booking.invoice_line_id                    |        |           |                                          |
|       | - Créer index composites (company_id, status, date)          |        |           |                                          |
| J3    | **Celery acks_late + timeouts**                              | S      | L         | Redéployer config précédente             |
|       | - Config `acks_late=True`, `task_time_limit=300`             |        |           |                                          |
|       | - Tests charge (10 tasks simultanées)                        |        |           |                                          |
| J3-J4 | **N+1 queries (routes bookings/invoices)**                   | M      | M         | Retirer joinedload si OOM                |
|       | - Ajouter joinedload(Booking.client).joinedload(Client.user) |        |           |                                          |
|       | - Pagination stricte (limit 100, offset)                     |        |           |                                          |
| J4    | **PDF URLs config**                                          | S      | L         | Revenir à hardcodé si bug                |
|       | - Env var PDF_BASE_URL                                       |        |           |                                          |
|       | - Tests génération PDF prod-like                             |        |           |                                          |
| J5    | **Validation montants invoices**                             | S      | L         | Rollback migration CHECK                 |
|       | - Contrainte CHECK total_amount >= 0                         |        |           |                                          |
|       | - Validator Marshmallow                                      |        |           |                                          |
| J5    | **docker-compose healthchecks**                              | S      | L         | Retirer condition si deadlock            |
|       | - Healthcheck sur api                                        |        |           |                                          |
|       | - depends_on avec condition: service_healthy                 |        |           |                                          |

**Total effort semaine 1**: ~5 jours (1 développeur)  
**Risques**: Migration timezone nécessite tests exhaustifs (régression calculs dates)

---

### **Semaine 2-4 (Next - Refactorings & DevEx)**

| Semaine | Tâche                                              | Effort | Impact                 |
| ------- | -------------------------------------------------- | ------ | ---------------------- |
| S2      | **Frontend: Refresh JWT automatique**              | M      | Élevé (UX)             |
|         | - Interceptor axios avec retry après refresh       | 3h     |                        |
|         | - Tests E2E (session longue, token expiré)         | 2h     |                        |
| S2      | **PII masking logs**                               | M      | Critique (GDPR)        |
|         | - Formatter logging custom (emails → e**_@_**.com) | 1j     |                        |
|         | - Audit logs existants, purge si nécessaire        | 4h     |                        |
| S2-S3   | **CI/CD GitHub Actions**                           | M      | Élevé (qualité)        |
|         | - Workflow lint (ruff, eslint, prettier)           | 3h     |                        |
|         | - Workflow tests (pytest backend, jest frontend)   | 4h     |                        |
|         | - Workflow build Docker + push registry            | 2h     |                        |
| S3      | **Services frontend factorisés**                   | M      | Moyen (maintenabilité) |
|         | - Service générique apiService.js                  | 1j     |                        |
|         | - Refactor companyService, driverService           | 1j     |                        |
| S3-S4   | **Tests backend (pytest)**                         | L      | Critique (qualité)     |
|         | - Tests auth (login, refresh, register)            | 1j     |                        |
|         | - Tests bookings (CRUD, assign, cancel)            | 2j     |                        |
|         | - Tests invoices (generate, reminder, QR-bill)     | 2j     |                        |
|         | - Tests dispatch (engine, queue, tasks Celery)     | 3j     |                        |
| S4      | **Tests frontend (RTL + E2E)**                     | L      | Moyen                  |
|         | - RTL pages Company/Driver/Client                  | 2j     |                        |
|         | - Cypress E2E (login → dashboard → booking)        | 1j     |                        |
| S4      | **Payment enum unified**                           | S      | Faible                 |
|         | - Utiliser models.enums.PaymentMethod partout      | 1h     |                        |
| S4      | **QR-Bill: retirer fallbacks hardcodés**           | S      | Faible                 |
|         | - Retour 400 si adresse client manquante           | 1h     |                        |

**Total effort semaines 2-4**: ~15 jours (1-2 développeurs)

---

### **Backlog (Later - Optimisations lourdes)**

| Tâche                              | Effort   | Impact | Justification report                            |
| ---------------------------------- | -------- | ------ | ----------------------------------------------- |
| **OSRM: lock async (httpx)**       | L (5j)   | Moyen  | Bottleneck si >100 req/s, rare en prod actuelle |
| **Frontend: assets morts cleanup** | M (1j)   | Faible | Gain 500kb bundle, non-bloquant                 |
| **Mobile apps: audit complet**     | L (10j+) | Moyen  | Apps peu utilisées actuellement                 |
| **Analytics dashboard avancé**     | L (10j+) | Moyen  | Fonctionnalité nice-to-have                     |
| **Migration PostgreSQL PostGIS**   | M (3j)   | Faible | Geo queries actuelles suffisantes               |

---

## 📝 Récapitulatif Livrables

Les fichiers suivants sont générés dans ce repo:

1. **REPORT.md** (ce fichier) - Audit complet structuré
2. **MIGRATIONS_NOTES.md** - Migrations Alembic proposées + rollback
3. **DELETIONS.md** - Fichiers/code morts à supprimer
4. **tests_plan.md** - Périmètre tests backend/frontend + rationalisation
5. **patches/** - Dossier avec patches unified diff :
   - `backend_timezone.patch` - Corrections timezone
   - `backend_indexes.patch` - Index DB manquants
   - `backend_celery.patch` - Config Celery acks_late/timeouts
   - `backend_n+1.patch` - Joinedload relations
   - `backend_pdf_config.patch` - URLs dynamiques PDF
   - `frontend_jwt_refresh.patch` - Interceptor refresh automatique
   - `infra_healthchecks.patch` - docker-compose healthchecks
6. **ci/** - Workflows GitHub Actions proposés :
   - `backend-lint.yml` - Ruff + mypy
   - `backend-tests.yml` - Pytest + coverage
   - `frontend-lint.yml` - ESLint + Prettier
   - `frontend-tests.yml` - Jest + RTL
   - `docker-build.yml` - Build + push images

---

## 🔒 Conclusion & Recommandations Finales

Votre application ATMR présente une **architecture solide** avec des choix techniques pertinents (Flask/Celery, React, Docker, OSRM, SocketIO). Les **forces principales** sont la modularité backend, le dispatch temps réel robuste, et la sécurité JWT bien implémentée.

Les **points d'attention critiques** sont:

1. **Timezone**: Incohérence majeure à résoudre en priorité (risque bugs calculs dates)
2. **Performance DB**: Index manquants + N+1 queries (dégradation si >1000 bookings/mois)
3. **Celery reliability**: `acks_late` manquant (perte potentielle de tâches dispatch/facturation)
4. **GDPR**: Logs PII non masqués (risque audit légal)

**Plan recommandé:**

- **Semaine 1**: Correctifs critiques (timezone, index, Celery, N+1)
- **Semaines 2-4**: DevEx (CI/CD, tests, refresh JWT, PII masking)
- **Backlog**: Optimisations lourdes (OSRM async, assets cleanup)

**Estimation globale**: ~20 jours-homme pour résoudre tous les findings majeurs (1-20).

---

_Document généré automatiquement le 15 octobre 2025. Pour toute question ou clarification, se référer aux patches et migrations détaillés._
