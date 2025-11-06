# Commandes Docker pour exécuter les tests E2E terminés

## Analyse des tests E2E terminés

D'après `backend/tests/TODO.md`, les tests E2E suivants sont terminés (✅ **TERMINÉ**) :

### Tests terminés (18 tests au total)
1. `test_update_booking_valid_schema`
2. `test_update_booking_invalid_schema`
3. `test_create_manual_booking_valid_schema`
4. `test_create_manual_booking_invalid_schema`
5. `test_create_client_valid_schema_self_service`
6. `test_create_client_valid_schema_private`
7. `test_create_client_valid_schema_corporate`
8. `test_create_client_invalid_schema`
9. `test_create_payment_valid_schema`
10. `test_create_payment_invalid_schema`
11. `test_update_client_valid_schema`
12. `test_update_client_invalid_schema`
13. `test_update_driver_profile_valid_schema`
14. `test_update_driver_profile_invalid_schema`
15. `test_update_billing_settings_valid_schema`
16. `test_update_billing_settings_invalid_schema`
17. `test_generate_invoice_valid_schema`
18. `test_generate_invoice_invalid_schema`
19. `test_update_payment_status_valid_schema`
20. `test_update_payment_status_invalid_schema`

---

## Commandes Docker recommandées

### Option 1 : Commande avec filtre `-k` (Recommandée)

C'est la méthode la plus précise, elle filtre uniquement les tests terminés :

```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -k "test_update_booking_valid_schema or test_update_booking_invalid_schema or \
      test_create_manual_booking_valid_schema or test_create_manual_booking_invalid_schema or \
      test_create_client_valid_schema_self_service or test_create_client_valid_schema_private or \
      test_create_client_valid_schema_corporate or test_create_client_invalid_schema or \
      test_create_payment_valid_schema or test_create_payment_invalid_schema or \
      test_update_client_valid_schema or test_update_client_invalid_schema or \
      test_update_driver_profile_valid_schema or test_update_driver_profile_invalid_schema or \
      test_update_billing_settings_valid_schema or test_update_billing_settings_invalid_schema or \
      test_generate_invoice_valid_schema or test_generate_invoice_invalid_schema or \
      test_update_payment_status_valid_schema or test_update_payment_status_invalid_schema" \
  -v --tb=short
```

### Option 2 : Pattern simple avec `-k`

Alternative plus simple utilisant un pattern pour sélectionner tous les tests avec `valid_schema` ou `invalid_schema` dans le nom :

```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -k "valid_schema or invalid_schema" \
  -v --tb=short
```

**Note** : Cette option pourrait inclure d'autres tests non terminés si leur nom contient `valid_schema` ou `invalid_schema`. Utilisez Option 1 pour être sûr.

### Option 3 : Tous les tests de la classe

Lance tous les tests de la classe `TestSchemaValidationE2E` (y compris ceux non terminés) :

```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -v --tb=short
```

### Option 4 : Tests spécifiques par groupe de fonctionnalités

#### Tests Bookings
```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -k "test_update_booking or test_create_manual_booking" \
  -v --tb=short
```

#### Tests Clients
```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -k "test_create_client or test_update_client" \
  -v --tb=short
```

#### Tests Payments
```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -k "test_create_payment or test_update_payment_status" \
  -v --tb=short
```

#### Tests Driver Profile
```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -k "test_update_driver_profile" \
  -v --tb=short
```

#### Tests Billing & Invoices
```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -k "test_update_billing_settings or test_generate_invoice" \
  -v --tb=short
```

### Option 5 : Test unique spécifique

Pour exécuter un seul test (exemple : DriverProfileUpdateSchema) :

```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_update_driver_profile_valid_schema \
  tests/e2e/test_schema_validation.py::TestSchemaValidationE2E::test_update_driver_profile_invalid_schema \
  -v --tb=short
```

---

## Options de sortie recommandées

### Format court (défaut)
- `-v` : Mode verbeux
- `--tb=short` : Traceback court en cas d'erreur

### Format détaillé
```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -k "... (filtre)" \
  -v -s --tb=long
```

### Avec couverture de code
```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -k "... (filtre)" \
  --cov=. --cov-report=term-missing \
  -v --tb=short
```

### Format JUnit XML (pour CI/CD)
```bash
docker-compose exec api pytest tests/e2e/test_schema_validation.py::TestSchemaValidationE2E \
  -k "... (filtre)" \
  --junitxml=test-results-e2e.xml \
  -v --tb=short
```

---

## Recommandation finale

**Utilisez l'Option 1** car elle :
- ✅ Sélectionne uniquement les tests terminés (validés dans TODO.md)
- ✅ Évite d'exécuter des tests non terminés ou en cours
- ✅ Est la plus précise et maintenable
- ✅ Facilite le débogage en cas d'échec

---

## Validation du résultat attendu

Une fois les tests exécutés, vous devriez voir :
- **18 tests** au total (ou 20 si vous incluez tous les tests valid_schema/invalid_schema)
- Tous marqués comme `PASSED` ✅
- Temps d'exécution affiché
- Résumé final avec nombre de tests passés/échoués

