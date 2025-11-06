# Synchronisation OpenAPI/Swagger avec Schemas Marshmallow

## ✅ Endpoints avec modèles Swagger synchronisés

### Auth

- ✅ `POST /api/auth/login` - `login_model` ↔ `LoginSchema`
- ✅ `POST /api/auth/register` - `register_model` ↔ `RegisterSchema`

### Bookings

- ✅ `POST /api/clients/<id>/bookings` - `booking_create_model` ↔ `BookingCreateSchema`
- ⚠️ `PUT /api/bookings/<id>` - Aucun modèle Swagger défini, utiliser `BookingUpdateSchema`
- ⚠️ `GET /api/bookings` - Query params documentés avec `@param`, correspond à `BookingListSchema`

### Companies

- ✅ `PUT /api/companies/me` - `company_update_model` ↔ `CompanyUpdateSchema`
- ✅ `POST /api/companies/me/drivers/create` - `create_driver_model` ↔ `DriverCreateSchema`
- ✅ `POST /api/companies/me/reservations/manual` - `manual_booking_model` ↔ `ManualBookingCreateSchema`
- ⚠️ `POST /api/companies/me/clients` - Aucun modèle Swagger défini, utiliser `ClientCreateSchema`

### Clients

- ⚠️ `PUT /api/clients/<id>` - `client_profile_model` existe, vérifier synchronisation avec `ClientUpdateSchema`

### Driver

- ⚠️ `PUT /api/driver/me/profile` - Aucun modèle Swagger défini, utiliser `DriverProfileUpdateSchema`

### Invoices

- ⚠️ `PUT /api/invoices/companies/<id>/billing-settings` - Aucun modèle Swagger défini, utiliser `BillingSettingsUpdateSchema`
- ⚠️ `POST /api/invoices/companies/<id>/invoices/generate` - Aucun modèle Swagger défini, utiliser `InvoiceGenerateSchema`

### Payments

- ✅ `PUT /api/payments/<id>` - `payment_status_model` existe, vérifier avec `PaymentStatusUpdateSchema`
- ✅ `POST /api/payments` - `payment_create_model` existe, vérifier avec `PaymentCreateSchema`

### Medical

- ✅ `GET /api/medical/establishments` - `@medical_ns.expect(establishments_parser)` ↔ `MedicalEstablishmentQuerySchema`
- ✅ `GET /api/medical/services` - `@medical_ns.expect(services_parser)` ↔ `MedicalServiceQuerySchema`

### Analytics

- ⚠️ `GET /api/analytics/dashboard` - Documenter avec `@param` ou modèle, correspond à `AnalyticsDashboardQuerySchema`
- ⚠️ `GET /api/analytics/insights` - Documenter avec `@param`, correspond à `AnalyticsInsightsQuerySchema`
- ⚠️ `GET /api/analytics/weekly-summary` - Documenter avec `@param`, correspond à `AnalyticsWeeklySummaryQuerySchema`
- ⚠️ `GET /api/analytics/export` - Documenter avec `@param`, correspond à `AnalyticsExportQuerySchema`

### Planning

- ⚠️ `GET /api/planning/companies/me/planning/shifts` - Documenter avec `@param`, correspond à `PlanningShiftsQuerySchema`
- ⚠️ `GET /api/planning/companies/me/planning/unavailability` - Documenter avec `@param`, correspond à `PlanningUnavailabilityQuerySchema`
- ⚠️ `GET /api/planning/companies/me/planning/weekly-template` - Documenter avec `@param`, correspond à `PlanningWeeklyTemplateQuerySchema`

### Admin

- ⚠️ `PUT /api/admin/users/<id>/role` - Aucun modèle Swagger défini, utiliser `UserRoleUpdateSchema`
- ⚠️ `POST /api/admin/autonomous-actions/<id>/review` - Aucun modèle Swagger défini, utiliser `AutonomousActionReviewSchema`

## Actions recommandées

1. **Créer modèles Swagger manquants** pour les endpoints POST/PUT qui n'en ont pas
2. **Ajouter `@param` pour query params** sur les endpoints GET Analytics/Planning
3. **Vérifier synchronisation** des modèles existants avec les schemas Marshmallow
4. **Documenter constraints de validation** (min/max length, regex, enum values) dans les modèles Swagger
