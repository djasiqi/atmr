# TODO List - Tâches par Priorité

## 🔴 PRIORITÉ CRITIQUE (P0) - Bloquant pour production

### Sécurité & Stabilité

#### 1. Migration des données chiffrées (Sécurité critique)

- [x] Migrer les données en clair vers les colonnes chiffrées ✅ **TERMINÉ**
  - Script : `backend/scripts/migrate_to_encryption.py`
  - Tables concernées : `user`, `client`
  - Commande : `python -m scripts.migrate_to_encryption [--dry-run]`
  - **Résultat** : Tous les utilisateurs et clients migrés (100%)
  - **Date** : 2025-10-29

#### 2. Corrections bugs/intégration Schemas (Stabilité API)

- [x] **Corriger intégration `PaymentCreateSchema` dans `backend/routes/payments.py`** ✅ **TERMINÉ**
  - Fichier: `backend/routes/payments.py`
  - Ligne: ~180 (méthode `post` de `CreatePayment`)
  - Remplacer validation manuelle par `validate_request(PaymentCreateSchema(), data)`
  - **Résultat** : Validation Marshmallow intégrée avec PaymentCreateSchema
  - **Date** : 2025-10-29
- [x] **Vérifier utilisation complète `validated_args` dans Analytics Export** ✅ **TERMINÉ**
  - Fichier: `backend/routes/analytics.py`
  - Ligne: ~242
  - Vérifier que tous les query params utilisent `validated_args` au lieu de `request.args`
  - **Résultat** : Tous les 4 endpoints utilisent déjà `validated_args` correctement
  - **Date** : 2025-10-29

#### 3. Migration DB - Table ProfilingMetrics

- [x] Appliquer la migration `3_4_add_profiling_metrics_table.py` ✅ **TERMINÉ**
  - Commande : `docker-compose exec api flask db upgrade`
  - Vérifier création table : `docker-compose exec postgres psql -U atmr -d atmr -c "\d profiling_metrics"`
  - **Résultat** : Table `profiling_metrics` créée avec succès, toutes les colonnes et index présents
  - **Révision Alembic** : `3_4_profiling` enregistrée
  - **Date** : 2025-10-29

---

## 🟠 PRIORITÉ HAUTE (P1) - Important pour qualité

### Tests Unitaires Schemas Marshmallow (Phase 2.4) - CRITIQUES

**Fichier cible**: `backend/tests/test_validation_schemas.py` ou fichiers dédiés

#### Schemas critiques (priorité haute - à faire en premier)

- [x] **Test `BookingUpdateSchema`** - Routes bookings ✅ **TERMINÉ** Test Pass

  - Validation mise à jour avec champs partiels ✅
  - Validation avec statut/dates invalides ✅
  - Validation montant négatif ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestBookingUpdateSchema`
  - **Résultat** : 8 tests passent (champs partiels, statut invalide, dates invalides, montant négatif, longueurs, booléens)

- [x] **Test `PaymentCreateSchema`** - Corriger intégration route + tests ✅ **TERMINÉ** Test Pass

  - Validation création paiement valide ✅
  - Validation amount requis et > 0 ✅
  - Validation method requis ✅
  - Validation booking_id optionnel ✅ (booking_id n'est pas requis, c'est optionnel)
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestPaymentCreateSchema`
  - **Résultat** : 7 tests passent (création valide, champs requis, amount > 0, method longueur, booking_id optionnel, reference optionnelle)

- [x] **Test `ClientCreateSchema`** - Routes clients ✅ **TERMINÉ** Test Pass

  - Validation création selon client_type (SELF_SERVICE, PRIVATE, CORPORATE) ✅
  - Validation champs requis selon type ✅ (validés par schéma et route)
  - Validation email pour SELF_SERVICE ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestClientCreateSchema`
  - **Résultat** : 10 tests passent (3 types clients, champs requis, email, longueurs, coordonnées GPS)

#### Schemas importants (priorité moyenne)

- [x] **Test `ManualBookingCreateSchema`** ✅ **TERMINÉ** Test Pass

  - Validation création réservation manuelle avec champs requis ✅
  - Validation champs optionnels (round trip, billing, medical) ✅
  - Validation formats (datetime ISO 8601, email, coordonnées GPS) ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestManualBookingCreateSchema`
  - **Résultat** : 11 tests passent (création minimale/complète, champs requis, formats, round trip, billed_to_type, longueurs, coordonnées, client_id, amount, email)

- [x] **Test `BillingSettingsUpdateSchema`** ✅ **TERMINÉ** Test Pass

  - Validation mise à jour partielle/complète des paramètres de facturation ✅
  - Validation payment_terms_days (0-365 jours) ✅
  - Validation frais (overdue_fee, reminder1fee, reminder2fee, reminder3fee >= 0) ✅
  - Validation IBAN et QR IBAN (format regex) ✅
  - Validation longueurs champs (email_sender, invoice_number_format, etc.) ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestBillingSettingsUpdateSchema`
  - **Résultat** : 9 tests passent (mise à jour partielle/complète, payment_terms_days, frais, IBAN, longueurs, booléens, reminder_schedule_days, templates)

- [x] **Test `InvoiceGenerateSchema`** ✅ **TERMINÉ** Test Pass

  - Validation génération facture avec client_id ou client_ids ✅
  - Validation period_year (2000-2100) et period_month (1-12) requis ✅
  - Validation client_ids (liste avec au moins 1 élément) ✅
  - Validation bill_to_client_id, reservation_ids, client_reservations (optionnels) ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestInvoiceGenerateSchema`
  - **Résultat** : 13 tests passent (client_id/client_ids, period_year/month, champs requis, limites, validation minimale)

- [x] **Test `PaymentStatusUpdateSchema`** ✅ **TERMINÉ** Test Pass

  - Validation mise à jour statut valide (pending, completed, failed) ✅
  - Validation status requis ✅
  - Validation status invalide (enum) ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestPaymentStatusUpdateSchema`
  - **Résultat** : 4 tests passent (statuts valides, status requis, status invalide, casse)

- [x] **Test `MedicalEstablishmentQuerySchema`** ✅ **TERMINÉ** Test Pass

  - Validation query params q (max 200 caractères) et limit (1-25) ✅
  - Validation valeurs par défaut (limit=8) ✅
  - Validation q optionnel ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestMedicalEstablishmentQuerySchema`
  - **Résultat** : 5 tests passent (query avec q/limit, valeurs par défaut, longueur q, validation limit, q optionnel)

- [x] **Test `MedicalServiceQuerySchema`** ✅ **TERMINÉ** Test pass

  - Validation establishment_id requis et >= 1 ✅
  - Validation query params q (max 200 caractères) optionnel ✅
  - Validation requête complète avec establishment_id et q ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestMedicalServiceQuerySchema`
  - **Résultat** : 6 tests passent (requête complète, seulement establishment_id, establishment_id manquant, validation establishment_id, longueur q, q optionnel)

- [x] **Test `AnalyticsDashboardQuerySchema`** ✅ **TERMINÉ** Test Pass

  - Validation period (7d|30d|90d|1y, défaut: 30d) ✅
  - Validation start_date et end_date optionnels (format YYYY-MM-DD) ✅
  - Validation formats de dates invalides ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestAnalyticsDashboardQuerySchema`
  - **Résultat** : 8 tests passent (period par défaut, period valides/invalides, dates personnalisées, validation formats, dates optionnelles, combinaisons)

- [x] **Test `AnalyticsInsightsQuerySchema`** ✅ **TERMINÉ** Test Pass

  - Validation lookback_days (1-365, défaut: 30) ✅
  - Validation limites min/max ✅
  - Validation type (Int requis, rejet Float/String) ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestAnalyticsInsightsQuerySchema`
  - **Résultat** : 6 tests passent (défaut, valeurs valides, validation min/max, type, valeurs limites)

- [x] **Test `AnalyticsWeeklySummaryQuerySchema`** ✅ **TERMINÉ** Test Pass

  - Validation week_start optionnel (format YYYY-MM-DD) ✅
  - Validation formats de dates invalides ✅
  - Test cas limites (dates début/fin année, 29 février) ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestAnalyticsWeeklySummaryQuerySchema`
  - **Résultat** : 5 tests passent (week_start présent/absent, validation format, optionnel, cas limites)

- [x] **Test `AnalyticsExportQuerySchema`** ✅ **TERMINÉ** Test Pass

  - Validation start_date et end_date requis (format YYYY-MM-DD) ✅
  - Validation format (csv|json, défaut: csv) ✅
  - Validation formats de dates invalides ✅
  - Test cas limites (dates identiques, début/fin année) ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestAnalyticsExportQuerySchema`
  - **Résultat** : 9 tests passent (requête complète, format par défaut, champs requis, validation formats dates/format, formats valides, cas limites)

- [x] **Test `PlanningShiftsQuerySchema`** ✅ **TERMINÉ** Test Pass

  - Validation driver_id optionnel (>= 1 si fourni) ✅
  - Validation type (Int requis, rejet Float/String non numérique) ✅
  - Test avec/sans driver_id ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestPlanningShiftsQuerySchema`
  - **Résultat** : 5 tests passent (driver_id présent/absent, validation limites, optionnel, type)

- [x] **Test `PlanningUnavailabilityQuerySchema`** ✅ **TERMINÉ** Test Pass

  - Validation driver_id optionnel (>= 1 si fourni) ✅
  - Validation type (Int requis, rejet Float/String non numérique) ✅
  - Test avec/sans driver_id ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestPlanningUnavailabilityQuerySchema`
  - **Résultat** : 5 tests passent (driver_id présent/absent, validation limites, optionnel, type)

- [x] **Test `PlanningWeeklyTemplateQuerySchema`** ✅ **TERMINÉ** Test Pass

  - Validation driver_id optionnel (>= 1 si fourni) ✅
  - Validation type (Int requis, rejet Float/String non numérique) ✅
  - Test avec/sans driver_id ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestPlanningWeeklyTemplateQuerySchema`
  - **Résultat** : 5 tests passent (driver_id présent/absent, validation limites, optionnel, type)

- [x] **Test `UserRoleUpdateSchema`** ✅ **TERMINÉ** Test Pass

  - Validation role requis (admin|client|driver|company) ✅
  - Validation company_id optionnel (>= 1 si fourni) ✅
  - Validation company_name optionnel (1-200 caractères si fourni) ✅
  - Test toutes les combinaisons valides ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestUserRoleUpdateSchema`
  - **Résultat** : 10 tests passent (tous les rôles, avec company_id/company_name, role manquant/invalide, validation company_id/company_name, champs optionnels)

- [x] **Test `AutonomousActionReviewSchema`** ✅ **TERMINÉ** Test Pass

  - Validation notes optionnel (max 1000 caractères) ✅
  - Test avec/sans notes ✅
  - Test caractères spéciaux et Unicode ✅
  - Test notes multilignes ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/test_validation_schemas.py::TestAutonomousActionReviewSchema`
  - **Résultat** : 6 tests passent (notes présentes/absentes, validation longueur, chaîne vide, caractères spéciaux, multilignes)

### Tests E2E Validation Schemas - ENDPOINTS CRITIQUES

**Fichier cible**: `backend/tests/e2e/test_schema_validation.py`

#### Endpoints critiques (priorité haute)

- Test payload valide (tous les champs) ✅
- Test payload invalide (format date, statut invalide, amount négatif) ✅
- Vérification erreurs 400 détaillées ✅
- **Date** : 2025-10-29
- **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_update_booking_valid_schema` et `test_update_booking_invalid_schema`
- **Résultat** : 2 tests E2E créés (validation succès, validation erreurs avec vérification messages d'erreur)

- [x] **Test E2E POST /api/companies/me/reservations/manual (`ManualBookingCreateSchema`)** ✅ **TERMINÉ**

  - Test payload valide (client_id, pickup/dropoff, scheduled_time, champs optionnels) ✅
  - Test payload invalide (client_id manquant, format date invalide, billed_to_type invalide, pickup_location trop long, amount négatif) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_create_manual_booking_valid_schema` et `test_create_manual_booking_invalid_schema`
  - **Résultat** : 2 tests E2E créés (validation succès, validation erreurs avec vérification messages d'erreur)

- [x] **Test E2E POST /api/companies/me/clients (`ClientCreateSchema`)** ✅ **TERMINÉ**

  - Test payload valide pour SELF_SERVICE (email requis) ✅
  - Test payload valide pour PRIVATE (first_name, last_name, address requis) ✅
  - Test payload valide pour CORPORATE (first_name, last_name, address requis) ✅
  - Test payload invalide (client_type manquant/invalide, email invalide, champs manquants selon type, limites longueur, coordonnées hors limites) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_create_client_valid_schema_*` et `test_create_client_invalid_schema`
  - **Résultat** : 4 tests E2E créés (3 pour types valides, 1 pour validations erreurs)

- [x] **Test E2E POST /api/payments/booking/<id> (`PaymentCreateSchema`)** ✅ **TERMINÉ**

  - Test payload valide (amount, method requis, reference optionnel) ✅
  - Test payload invalide (amount/method manquants, amount < 0.01, amount négatif, method/reference trop longs) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_create_payment_valid_schema` et `test_create_payment_invalid_schema`
  - **Résultat** : 2 tests E2E créés (validation succès, validation erreurs avec vérification messages d'erreur)

#### Endpoints importants

- [x] **Test E2E PUT /api/clients/<id> (`ClientUpdateSchema`)** ✅ **TERMINÉ**

  - Test payload valide (tous les champs optionnels, mise à jour complète/partielle/vide) ✅
  - Test payload invalide (first_name/last_name trop longs, phone invalide/trop court, address trop long, birth_date format invalide, gender invalide) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_update_client_valid_schema` et `test_update_client_invalid_schema`
  - **Résultat** : 2 tests E2E créés (validation succès avec différents scénarios, validation erreurs avec vérification messages d'erreur)

- [x] **Test E2E PUT /api/driver/me/profile (`DriverProfileUpdateSchema`)** ✅ **TERMINÉ**

  - Test payload valide (tous les champs optionnels, mise à jour complète/partielle/vide) ✅
  - Test payload invalide (first_name/last_name trop longs, status invalide, weekly_hours hors limites, hourly_rate_cents négatif, dates format invalide, license_categories/trainings trop nombreuses) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_update_driver_profile_valid_schema` et `test_update_driver_profile_invalid_schema`
  - **Résultat** : 2 tests E2E créés (validation succès avec différents scénarios, validation erreurs avec vérification messages d'erreur)

- [x] **Test E2E PUT /api/invoices/companies/<id>/billing-settings (`BillingSettingsUpdateSchema`)** ✅ **TERMINÉ**

  - Test payload valide (tous les champs optionnels, mise à jour complète/partielle/vide) ✅
  - Test payload invalide (payment_terms_days hors limites, fees négatifs, email/format/prefix/templates trop longs, IBAN invalide, esr_ref_base trop long, pdf_template_variant trop long) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_update_billing_settings_valid_schema` et `test_update_billing_settings_invalid_schema`
  - **Résultat** : 2 tests E2E créés (validation succès avec différents scénarios, validation erreurs avec vérification messages d'erreur)

- [x] **Test E2E POST /api/invoices/companies/<id>/invoices/generate (`InvoiceGenerateSchema`)** ✅ **TERMINÉ**

  - Test payload valide (client_id simple, client_ids groupé, bill_to_client_id facturation tierce, reservation_ids, client_reservations) ✅
  - Test payload invalide (period_year/month manquants/hors limites, client_id/bill_to_client_id négatifs, client_ids vide, absence client_id/client_ids) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_generate_invoice_valid_schema` et `test_generate_invoice_invalid_schema`
  - **Résultat** : 2 tests E2E créés (validation succès avec différents scénarios, validation erreurs avec vérification messages d'erreur)

- [x] **Test E2E PUT /api/payments/<id> (`PaymentStatusUpdateSchema`)** ✅ **TERMINÉ**

  - Test payload valide (status: "pending", "completed", "failed") ✅
  - Test payload invalide (status manquant, status invalide, status en majuscules, status vide) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-10-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_update_payment_status_valid_schema` et `test_update_payment_status_invalid_schema`
  - **Résultat** : 2 tests E2E créés (validation succès avec les 3 statuts valides, validation erreurs avec vérification messages d'erreur)

- [x] **Test E2E GET /api/analytics/weekly-summary (`AnalyticsWeeklySummaryQuerySchema`)** ✅ **TERMINÉ**

  - Test query params valides (week_start spécifié et week_start optionnel) ✅
  - Test query params invalides (format date invalide, date mal formée) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-01-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_analytics_weekly_summary_valid_query` et `test_analytics_weekly_summary_invalid_query`
  - **Résultat** : 2 tests E2E créés (validation succès avec/sans week_start, validation erreurs avec vérification messages d'erreur)

- [x] **Test E2E GET /api/planning/companies/me/planning/unavailability (`PlanningUnavailabilityQuerySchema`)** ✅ **TERMINÉ**

  - Test query params valides (driver_id spécifié et driver_id optionnel) ✅
  - Test query params invalides (driver_id négatif, driver_id = 0, driver_id non numérique) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-01-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_planning_unavailability_valid_query` et `test_planning_unavailability_invalid_query`
  - **Résultat** : 2 tests E2E créés (validation succès avec/sans driver_id, validation erreurs avec vérification messages d'erreur)

- [x] **Test E2E GET /api/planning/companies/me/planning/weekly-template (`PlanningWeeklyTemplateQuerySchema`)** ✅ **TERMINÉ**

  - Test query params valides (driver_id spécifié et driver_id optionnel) ✅
  - Test query params invalides (driver_id négatif, driver_id = 0, driver_id non numérique) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-01-29
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_planning_weekly_template_valid_query` et `test_planning_weekly_template_invalid_query`
  - **Résultat** : 2 tests E2E créés (validation succès avec/sans driver_id, validation erreurs avec vérification messages d'erreur)

- [x] **Test E2E POST /api/admin/autonomous-actions/<id>/review (`AutonomousActionReviewSchema`)** ✅ **TERMINÉ**

  - Test payload valide (notes optionnelles, marquage reviewed_by_admin) ✅
  - Test payload invalide (notes > 1000, schéma invalide) ✅
  - Vérification erreurs 400 détaillées ✅
  - **Date** : 2025-10-30
  - **Fichier** : `backend/tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_autonomous_action_review_valid` et `test_autonomous_action_review_invalid`
  - **Résultat** : 2 tests E2E créés et passants

### Augmenter couverture tests à 70% (3.1)

- [x] Générer couverture globale E2E (coverage.xml) ✅

  - Commande: `docker-compose run --rm -e FLASK_ENV=testing api pytest tests/e2e/test_schema_validation.py -k "update_booking or create_manual_booking or create_client_ or update_client_ or create_payment or update_payment_status or planning_ or analytics_ or medical_ or autonomous_action_review" --cov=. --cov-report=xml --cov-report=term-missing -q`
  - Résultat: Couverture globale actuelle: **28.85%** (objectif 70%)

- [x] Identifier modules prioritaires < 80% (critique) et 0% ✅

  - Critiques < 80% à prioriser (Routes): `routes/companies.py` (~33.57%), `routes/dispatch_routes.py` (~21.42%), `routes/bookings.py` (~33.33%), `routes/auth.py` (~42.39%), `routes/admin.py` (~33.82%), `routes/payments.py` (~50.00%)
  - Critiques (Models): `models/booking.py` (~57.14%), `models/client.py` (~40.00%), `models/driver.py` (~80.77%), `models/user.py` (~43.95%)
  - Critiques (Unified Dispatch): `services/unified_dispatch/heuristics.py` (~8.99%), `queue.py` (~57.22%), `realtime_optimizer.py` (~22.71%), `reactive_suggestions.py` (~23.62%)
  - Sécurité (0%): `security/crypto.py`, `security/audit_log.py`
  - Autres services à renforcer: `services/api_slo.py` (~49.15%), `middleware/metrics.py` (~66.67%)

- [ ] Monter à ≥ 70%: plan d’attaque par vagues (priorité décroissante)

  1. Routes: créer tests ciblés unitaires/fonctionnels sur `routes/companies.py`, `routes/dispatch_routes.py`, `routes/bookings.py`
  2. Models: ajouter tests de méthodes/validations sur `models/booking.py`, `models/client.py`, `models/user.py`
  3. Unified Dispatch: tests unitaires sur heuristics, queue, realtime_optimizer (mocker I/O et OSRM)
  4. Sécurité: tests de `security/crypto.py` (chiffrement/déchiffrement) et `security/audit_log.py`
  5. Compléments services: `services/api_slo.py`, `middleware/metrics.py`

- [ ] **Créer tests pour modules critiques < 80% (priorité haute)**

  - Routes API: `routes/bookings.py`, `routes/companies.py`, `routes/auth.py`, `routes/admin.py`, `routes/dispatch_routes.py`, `routes/payments.py`
  - Dispatch: `services/unified_dispatch/engine.py`, `solver.py`, `heuristics.py`, `autonomous_manager.py`, `queue.py`
  - Sécurité: `security/crypto.py`, `security/audit_log.py`
  - Services: `services/api_slo.py`, `services/unified_dispatch/slo.py`, `middleware/metrics.py`
  - Models: `models/booking.py`, `models/client.py`, `models/driver.py`, `models/user.py`

- [ ] **Créer tests pour modules non testés (0% couverture)**

  - Prioriser modules avec plus de lignes (> 50 lignes)

- [ ] **Maintenir couverture ≥ 70% dans chaque PR**

---

## 🟡 PRIORITÉ MOYENNE (P2) - Amélioration continue

### Monitoring & Observabilité

#### Dashboard Grafana SLO API (2.10)

- [ ] **Créer le fichier JSON du dashboard Grafana** (`grafana/dashboards/api_slo.json`)
- [ ] **Configurer Prometheus scraping** (vérifier `/prometheus/metrics-http`)
- [ ] **Configurer Grafana datasource** (Prometheus)
- [ ] **Créer des alertes Grafana** (optionnel mais recommandé)
- [ ] **Tests du dashboard** (générer données test, valider panels)
- [ ] **Documentation** (`grafana/dashboards/README.md`)
- [ ] **Sécurité et accès** (admin uniquement)
- [ ] **Intégration CI/CD** (validation JSON dashboard - optionnel)

#### Prometheus métriques avancées (2.10)

- [ ] **Ajouter dashboard Grafana pour métriques HTTP** (`grafana/dashboards/http_metrics.json`)
- [ ] **Configurer alertes personnalisées sur métriques HTTP** (`prometheus/alerts-http.yml`)
- [ ] **Exposer métriques métier supplémentaires**
  - `bookings_created_total`, `bookings_completed_total`, `bookings_active`
  - `booking_distance_meters`, `dispatch_triggered_total`, `db_query_duration_seconds`

#### Déploiement PagerDuty pour Alertes SLO (2.11)

- [ ] **Déployer Alertmanager** (docker-compose ou Kubernetes)
- [ ] **Configurer la clé PagerDuty dans alertmanager.yml**
- [ ] **Tester les alertes en production**
- [ ] **Créer les runbooks référencés dans les annotations**
  - `/runbooks/api-slo-latency.md`
  - `/runbooks/api-slo-error-rate.md`
  - `/runbooks/api-slo-availability.md`
  - `/runbooks/dispatch-slo-breach.md`
  - `/runbooks/dispatch-slo-critical.md`
  - `/runbooks/health-check-failure.md`
  - `/runbooks/global-slo-summary.md`

### API Versioning (3.2)

- [ ] **Tester les routes v1 en production** (vérifier headers Deprecation)
- [ ] **Tester les routes v2** (peut être vide pour l'instant)
- [ ] **Migrer frontend vers /api/v1/** (remplacer `/api/` par `/api/v1/`)
- [ ] **Créer premières routes v2** (bookings, companies, auth)
- [ ] **Tests E2E versioning** (headers, migration progressive, désactivation legacy)

### Tests Chaos Engineering (D3)

- [ ] **Tester fallback haversine quand OSRM down** (chaos injector)
- [ ] **Tester DB read-only avec vraies routes API** (POST /api/bookings)

---

## 🟢 PRIORITÉ BASSE (P3) - Nice to have

### Documentation & Optimisation

- [ ] **Documentation supplémentaire** (runbooks détaillés, guides avancés)
- [ ] **Optimisations performance** (basées sur profiling 3.4)
- [ ] **Tests E2E versioning avancés** (après migration complète)

---

## 📝 Notes

- **Commandes utiles** : Voir sections détaillées ci-dessous pour chaque catégorie
- **Critères d'acceptation** : Définis dans chaque section
- **Fichiers créés/modifiés** : Référencés dans chaque section

---

## 🔗 Sections détaillées (référence)

[Les sections détaillées existantes peuvent être conservées en bas du fichier pour référence complète]
