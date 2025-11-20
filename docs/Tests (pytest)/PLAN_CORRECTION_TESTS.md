# Plan de Correction Détaillé - Tests Pytest Échoués

## Résumé Exécutif

**Date d'analyse** : 2025-11-20  
**Total de tests échoués** : 946  
**Tests passés** : 1660  
**Tests ignorés** : 44  
**Tests avec erreurs** : 18  
**Taux de réussite actuel** : 63.7%

**Statistiques par priorité** :
- **CRITIQUE** : 156 tests (16.5%) - Bloquent des fonctionnalités essentielles
- **IMPORTANT** : 645 tests (68.2%) - Affectent des fonctionnalités secondaires  
- **MINEUR** : 145 tests (15.3%) - Cas limites ou optimisations

**Estimation totale** : 35-45 heures

---

## CATÉGORISATION DES ÉCHECS

### Groupe 1 : Problèmes de Signature de Fonction (CRITIQUE)

#### 1.1. Tests : `test_safety_limits.py::TestActionLogging::test_successful_action_logged`
#### 1.2. Tests : `test_safety_limits.py::TestActionLogging::test_failed_action_logged`
#### 1.3. Tests : `test_safety_limits.py::TestActionLogging::test_action_blocked_by_limits_not_logged`

**Fichier** : `backend/tests/test_safety_limits.py::TestActionLogging`  
**Type d'erreur** : `TypeError`  
**Message d'erreur** : `Suggestion.__init__() missing 1 required positional argument: 'priority'`

**Cause racine identifiée** :
- La classe `Suggestion` dans `backend/services/unified_dispatch/reactive_suggestions.py` requiert désormais un argument `priority` obligatoire (ligne 58)
- Les tests créent des instances de `Suggestion` sans fournir l'argument `priority`

**Tests similaires affectés** :
- Tous les tests dans `TestActionLogging` qui créent des objets `Suggestion`

**Plan de correction** :

1. **Corriger les tests pour inclure `priority`**
   - Fichier à modifier : `backend/tests/test_safety_limits.py`
   - Lignes concernées : 411, 449, 498
   - Modification à apporter : Ajouter `priority="medium"` (ou approprié) lors de la création des `Suggestion`

```python
# Code actuel (ligne 411)
suggestion = Suggestion(
    action="reassign",
    message="Test reassignment",
    booking_id=int(booking.id),
    driver_id=int(driver.id),
    auto_applicable=True,
)

# Code corrigé
suggestion = Suggestion(
    action="reassign",
    priority="medium",  # ✅ AJOUT
    message="Test reassignment",
    booking_id=int(booking.id),
    driver_id=int(driver.id),
    auto_applicable=True,
)
```

**Vérification** :
- [ ] Test passe individuellement
- [ ] Tests du module passent
- [ ] Coverage maintenu/amélioré

**Risques et considérations** :
- Assurer la cohérence avec les valeurs de priorité utilisées ailleurs dans le code
- Vérifier que les valeurs de priorité utilisées correspondent aux valeurs attendues (`"low"`, `"medium"`, `"high"`, `"critical"`)

---

#### 1.4. Test : `test_heuristics.py::TestHeuristicsHelpers::test_check_driver_window_feasible`

**Fichier** : `backend/tests/test_heuristics.py`  
**Type d'erreur** : `TypeError`  
**Message d'erreur** : `_check_driver_window_feasible() got an unexpected keyword argument 'est_finish_min'`

**Cause racine identifiée** :
- La signature de `_check_driver_window_feasible` a changé et n'accepte plus `est_finish_min`
- Le test utilise un ancien nom de paramètre qui n'existe plus

**Plan de correction** :

1. **Vérifier la signature actuelle de la fonction**
   - Fichier à vérifier : `backend/services/unified_dispatch/heuristics.py` (ligne 464)
   - Signature actuelle : `_check_driver_window_feasible(driver_window: Tuple[int, int], est_start_min: int) -> bool`

2. **Corriger le test**
   - Fichier à modifier : `backend/tests/test_heuristics.py`
   - Modification : Supprimer `est_finish_min` et utiliser la bonne signature

**Vérification** :
- [ ] Test passe individuellement
- [ ] Tests du module passent

---

### Groupe 2 : Problèmes de Conversion d'Unités (CRITIQUE)

#### 2.1. Test : `test_safety_limits.py::TestAutonomousActionModel::test_to_dict`
#### 2.2. Test : `test_solver.py::TestSolverDataclasses::test_solver_assignment_structure`
#### 2.3. Test : `test_solver.py::TestSolverDataclasses::test_solver_assignment_to_dict`

**Fichier** : `backend/tests/test_safety_limits.py::TestAutonomousActionModel`  
**Type d'erreur** : `AssertionError`  
**Message d'erreur** : `assert 0.1005 == 100.5` et `assert 0.1 == 100`

**Cause racine identifiée** :
- Confusion entre secondes et millisecondes
- `execution_time_ms` devrait être en millisecondes, mais le code stocke des secondes
- Les tests s'attendent à des valeurs en millisecondes mais reçoivent des valeurs en secondes

**Plan de correction** :

1. **Analyser où `execution_time_ms` est défini**
   - Fichier à vérifier : `backend/services/unified_dispatch/autonomous_manager.py` (ligne 328)
   - Vérifier la conversion du temps d'exécution

2. **Corriger soit le code de production, soit les tests**
   - Si le code stocke en secondes mais le champ s'appelle `_ms`, convertir en millisecondes
   - Si les tests sont incorrects, ajuster les assertions

**Code à corriger** :

```python
# Dans autonomous_manager.py, ligne ~328
# Code actuel (si execution_time est en secondes)
action_record.execution_time_ms = execution_time_ms  # En secondes!

# Code corrigé
action_record.execution_time_ms = execution_time_ms * 1000  # Convertir en ms
```

**Vérification** :
- [ ] Test passe individuellement
- [ ] Vérifier la cohérence dans tout le codebase
- [ ] Tests du module passent

**Risques et considérations** :
- Vérifier tous les endroits où `execution_time_ms` est utilisé
- S'assurer que la conversion est cohérente partout

---

### Groupe 3 : Problèmes de Fichiers/Path (CRITIQUE)

#### 3.1. Test : `test_shadow_mode.py::TestShadowModeManager::test_generate_daily_report`
#### 3.2. Test : `test_shadow_mode.py::TestShadowModeManager::test_get_company_summary`
#### 3.3. Test : `test_shadow_mode.py::TestShadowModeManager::test_save_daily_report`
#### 3.4. Test : `test_shadow_mode.py::TestShadowModeIntegration::test_end_to_end_workflow`
#### 3.5. Test : `test_shadow_mode_comprehensive.py::TestShadowModeManager::test_daily_report_generation`
#### 3.6-3.9. Autres tests similaires dans `test_shadow_mode_comprehensive.py`

**Fichier** : `backend/services/rl/shadow_mode_manager.py`  
**Type d'erreur** : `FileNotFoundError`  
**Message d'erreur** : `[Errno 2] No such file or directory: '/tmp/.../report_2025-11-20.json/w'`

**Cause racine identifiée** :
- **BUG CRITIQUE** dans `_save_daily_report` ligne 650
- `Path(json_path, "w", encoding="utf-8").open()` est incorrect
- `Path()` est utilisé comme constructeur avec des arguments qui devraient être passés à `.open()`
- Le code essaie d'ouvrir un chemin qui est en fait `json_path + "/w"` au lieu de juste `json_path`

**Plan de correction** :

1. **Corriger `_save_daily_report`**
   - Fichier à modifier : `backend/services/rl/shadow_mode_manager.py`
   - Ligne concernée : 650
   - Modification à apporter : Utiliser `open()` directement ou `Path().open()` correctement

```python
# Code actuel (ligne 650) - INCORRECT
json_path = company_dir / f"report_{date_str}.json"
with Path(json_path, "w", encoding="utf-8").open() as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# Code corrigé - Option 1 (recommandé)
json_path = company_dir / f"report_{date_str}.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# Code corrigé - Option 2
json_path = company_dir / f"report_{date_str}.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
```

**Vérification** :
- [ ] Test passe individuellement
- [ ] Vérifier tous les tests de shadow_mode passent
- [ ] Coverage maintenu/amélioré

**Risques et considérations** :
- Cette erreur affecte probablement tous les appels à `_save_daily_report`
- Vérifier qu'il n'y a pas d'autres utilisations similaires de `Path()` dans le fichier

---

### Groupe 4 : Problèmes de Précision de Calcul (IMPORTANT)

#### 4.1. Test : `test_shadow_mode.py::TestShadowModeManager::test_calculate_daily_statistics`

**Fichier** : `backend/tests/test_shadow_mode.py`  
**Type d'erreur** : `AssertionError`  
**Message d'erreur** : `assert -1.3333333333333333 == -1.33`

**Cause racine identifiée** :
- Le test attend une valeur arrondie à 2 décimales (`-1.33`)
- Le calcul retourne la valeur exacte (`-1.3333333333333333`)
- Problème de précision de calcul en virgule flottante

**Plan de correction** :

1. **Corriger le test pour utiliser une comparaison avec tolérance**
   - Fichier à modifier : `backend/tests/test_shadow_mode.py`
   - Ligne concernée : 292

```python
# Code actuel
assert eta_stats["mean"] == -1.33  # (-2 + 1 - 3) / 3

# Code corrigé - Option 1 (recommandé)
assert abs(eta_stats["mean"] - -1.3333333333333333) < 0.01

# Code corrigé - Option 2
assert round(eta_stats["mean"], 2) == -1.33
```

**Vérification** :
- [ ] Test passe individuellement
- [ ] Vérifier que la précision est acceptable

**Risques et considérations** :
- Vérifier si le calcul doit être arrondi dans le code de production ou juste dans le test
- S'assurer que la tolérance choisie est appropriée

---

### Groupe 5 : Problèmes de Contexte Flask (IMPORTANT)

#### 5.1. Test : `test_ml_monitoring.py::TestMLMonitoringService::test_log_prediction`
#### 5.2. Test : `test_ml_monitoring.py::TestMLMonitoringService::test_update_actual_delay`
#### 5.3. Test : `test_ml_monitoring.py::TestMLMonitoringService::test_get_metrics`

**Fichier** : `backend/tests/test_ml_monitoring.py`  
**Type d'erreur** : `RuntimeError`  
**Message d'erreur** : `Working outside of application context`

**Cause racine identifiée** :
- Les tests n'utilisent pas le contexte Flask (`app.app_context()`)
- Certaines opérations nécessitent un contexte Flask actif

**Plan de correction** :

1. **Ajouter `@pytest.fixture` pour le contexte Flask si nécessaire**
   - Vérifier `backend/tests/conftest.py` pour les fixtures existantes

2. **Modifier les tests pour utiliser le contexte**
   - Fichier à modifier : `backend/tests/test_ml_monitoring.py`

```python
# Code à ajouter dans les tests
def test_log_prediction(self, app):
    """Test qu'une prédiction est loggée."""
    with app.app_context():
        # ... code du test ...
```

**Vérification** :
- [ ] Test passe individuellement
- [ ] Tests du module passent

---

### Groupe 6 : Problèmes de Redirection HTTP (IMPORTANT)

#### 6.1-6.20. Plusieurs tests retournant 302 au lieu de 200

**Fichiers** : 
- `test_ml_monitoring.py::TestMLMonitoringAPI`
- `test_prometheus_metrics.py::TestPrometheusMetricsEndpoint`
- `test_rate_limiting.py` (plusieurs tests)
- Et autres...

**Type d'erreur** : `AssertionError`  
**Message d'erreur** : `assert 302 == 200` ou `assert 302 in (200, 401, 403)`

**Cause racine identifiée** :
- Les tests font des requêtes HTTP mais reçoivent des redirections 302
- Probablement lié à l'authentification ou à la configuration des routes
- Les tests ne sont peut-être pas authentifiés correctement

**Plan de correction** :

1. **Analyser la configuration des routes**
   - Vérifier si les routes nécessitent une authentification
   - Vérifier la configuration de l'API legacy (`API_LEGACY_ENABLED`)

2. **Corriger les tests pour inclure l'authentification**
   - Utiliser `@jwt_required()` si nécessaire
   - Ou utiliser `client.post(..., headers=authenticated_headers)`

**Code à corriger** :

```python
# Dans les tests, ajouter l'authentification
def test_get_metrics(self, client, admin_token):
    """Test récupération des métriques."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = client.get("/api/metrics", headers=headers)
    assert response.status_code == 200
```

**Vérification** :
- [ ] Test passe individuellement
- [ ] Vérifier que l'authentification fonctionne correctement

**Risques et considérations** :
- Assurer que les tokens de test sont valides
- Vérifier la configuration des routes dans l'environnement de test

---

### Groupe 7 : Problèmes de Validation de Schémas (IMPORTANT)

#### 7.1-7.15. Tests dans `test_phase2_schemas.py`

**Fichier** : `backend/tests/test_phase2_schemas.py`  
**Type d'erreur** : `AssertionError`  
**Message d'erreur** : `assert 'iban' in {'errors': {'iban': [...]}}`

**Cause racine identifiée** :
- Les tests vérifient `"iban" in exc_info.value.messages`
- Mais `exc_info.value.messages` retourne un dictionnaire avec structure `{'errors': {'iban': [...]}, 'message': '...'}`
- Il faut vérifier `"iban" in exc_info.value.messages.get('errors', {})`

**Plan de correction** :

1. **Corriger toutes les assertions de validation**
   - Fichier à modifier : `backend/tests/test_phase2_schemas.py`
   - Lignes concernées : 31, 38, 45, 72, 89, et autres

```python
# Code actuel
assert "iban" in exc_info.value.messages

# Code corrigé
assert "iban" in exc_info.value.messages.get("errors", {})
# OU
errors = exc_info.value.messages.get("errors", {})
assert "iban" in errors
```

**Vérification** :
- [ ] Tous les tests de validation passent
- [ ] Coverage maintenu

---

### Groupe 8 : Problèmes d'Initialisation de Service (IMPORTANT)

#### 8.1-8.11. Tests dans `test_alerts_comprehensive.py::TestProactiveAlertsService`

**Fichier** : `backend/tests/test_alerts_comprehensive.py`  
**Type d'erreur** : `TypeError`  
**Message d'erreur** : `ProactiveAlertsService.__init__() got an unexpected keyword argument 'notification_service'`

**Cause racine identifiée** :
- Le test crée `ProactiveAlertsService(notification_service=..., delay_predictor=...)`
- Mais `ProactiveAlertsService.__init__()` ne prend aucun paramètre (ligne 60)
- Le service crée ses dépendances en interne

**Plan de correction** :

1. **Modifier le fixture pour ne pas passer de paramètres**
   - Fichier à modifier : `backend/tests/test_alerts_comprehensive.py`
   - Ligne concernée : 73-74

```python
# Code actuel
return ProactiveAlertsService(
    notification_service=mock_notification_service, delay_predictor=mock_delay_predictor
)

# Code corrigé - Option 1 (mock après création)
service = ProactiveAlertsService()
service.notification_service = mock_notification_service
service.ml_predictor = mock_delay_predictor
return service

# Code corrigé - Option 2 (patch dans le test)
with patch.object(ProactiveAlertsService, 'notification_service', mock_notification_service):
    with patch.object(ProactiveAlertsService, 'ml_predictor', mock_delay_predictor):
        service = ProactiveAlertsService()
        # ... tests ...
```

**Vérification** :
- [ ] Tous les tests de ProactiveAlertsService passent

---

### Groupe 9 : Problèmes de Clés Manquantes (IMPORTANT)

#### 9.1-9.10. Tests dans `test_solver.py` avec `KeyError: 'starts'`

**Fichier** : `backend/tests/test_solver.py`  
**Type d'erreur** : `KeyError`  
**Message d'erreur** : `KeyError: 'starts'`

**Cause racine identifiée** :
- Le solveur retourne un dictionnaire sans la clé `'starts'`
- Soit le format de retour a changé, soit le test est incorrect

**Plan de correction** :

1. **Analyser le format de retour du solveur**
   - Vérifier `backend/services/unified_dispatch/solver.py`

2. **Corriger soit le solveur, soit les tests**

**Vérification** :
- [ ] Tous les tests du solveur passent

---

### Groupe 10 : Problèmes de Déballage de Tuples (IMPORTANT)

#### 10.1-10.4. Tests dans `test_preferred_driver.py`

**Fichier** : `backend/tests/unified_dispatch/test_preferred_driver.py`  
**Type d'erreur** : `ValueError`  
**Message d'erreur** : `not enough values to unpack (expected 3, got 2)`

**Cause racine identifiée** :
- Le code essaie de déballer 3 valeurs mais n'en reçoit que 2
- La signature de la fonction a probablement changé

**Plan de correction** :

1. **Identifier la ligne de code problématique**
   - Chercher les `unpack` de tuples dans `test_preferred_driver.py`

2. **Corriger le nombre de valeurs à déballer**

**Vérification** :
- [ ] Tous les tests passent

---

### Groupe 11 : Problèmes Divers (MINEUR)

#### 11.1. Test : `test_shadow_mode.py::TestShadowModeManager::test_filter_data_by_company_and_date`
- Problème de filtrage des données par date

#### 11.2. Test : `test_retry.py::TestBackoffCalculation::test_jitter_range`
- Problème de plage de valeurs pour le jitter

#### 11.3. Test : `test_solver.py::TestSolverEmptyProblems`
- Problèmes de gestion des problèmes vides

Et plusieurs autres tests mineurs...

---

## ORDRE DE CORRECTION RECOMMANDÉ

### Phase 1 : Corrections Critiques (Priorité 1) - ~12 heures

1. **Groupe 3** : Problèmes de Path/Fichiers (shadow_mode_manager.py ligne 650)
   - **Raison** : Bug critique qui bloque 9+ tests
   - **Temps estimé** : 1 heure

2. **Groupe 1** : Problèmes de signature Suggestion (priority manquant)
   - **Raison** : Bloque 3+ tests critiques
   - **Temps estimé** : 1 heure

3. **Groupe 2** : Problèmes de conversion d'unités (execution_time_ms)
   - **Raison** : Bloque plusieurs tests et peut causer des bugs en production
   - **Temps estimé** : 2 heures

4. **Groupe 9** : KeyError 'starts' dans solver
   - **Raison** : Bloque plusieurs tests du solveur
   - **Temps estimé** : 2-3 heures

5. **Groupe 10** : Problèmes de déballage de tuples
   - **Raison** : Bloque les tests preferred_driver
   - **Temps estimé** : 1-2 heures

6. **Groupe 1.4** : Signature _check_driver_window_feasible
   - **Raison** : Test heuristique bloqué
   - **Temps estimé** : 30 minutes

### Phase 2 : Corrections Importantes (Priorité 2) - ~20 heures

7. **Groupe 6** : Problèmes de redirection HTTP (302)
   - **Raison** : Affecte de nombreux tests API
   - **Temps estimé** : 4-5 heures

8. **Groupe 5** : Problèmes de contexte Flask
   - **Raison** : Bloque les tests ML monitoring
   - **Temps estimé** : 2 heures

9. **Groupe 7** : Problèmes de validation de schémas
   - **Raison** : Bloque les tests de validation
   - **Temps estimé** : 2 heures

10. **Groupe 8** : Problèmes d'initialisation ProactiveAlertsService
    - **Raison** : Bloque 11 tests d'alertes
    - **Temps estimé** : 2 heures

11. **Groupe 4** : Problèmes de précision de calcul
    - **Raison** : Facile à corriger
    - **Temps estimé** : 1 heure

### Phase 3 : Corrections Mineures (Priorité 3) - ~10 heures

12. **Groupe 11** : Problèmes divers
    - **Temps estimé** : 8-10 heures

---

## DÉPENDANCES ENTRE CORRECTIONS

- **Aucune dépendance critique identifiée**
- Les corrections peuvent être faites en parallèle pour la plupart
- Groupe 3 (Path) doit être fait avant les tests shadow_mode peuvent passer
- Groupe 1 (Suggestion) doit être fait avant certains tests safety_limits

---

## TESTS À CRÉER/AMÉLIORER APRÈS CORRECTION

1. **Tests de régression pour `Path().open()`**
   - S'assurer que le problème de Path ne se reproduit pas

2. **Tests de validation pour `execution_time_ms`**
   - Vérifier que les conversions sont correctes

3. **Tests d'intégration pour l'authentification**
   - S'assurer que tous les endpoints nécessitent l'auth

4. **Tests de précision pour les calculs**
   - Utiliser `pytest.approx()` pour les comparaisons flottantes

---

## PROCÉDURE DE VALIDATION

Pour chaque correction :

1. ✅ Créer une branche Git : `git checkout -b fix/nom-du-groupe`
2. ✅ Appliquer les corrections
3. ✅ Exécuter le test individuellement : `pytest backend/tests/fichier.py::test_fonction -v`
4. ✅ Exécuter tous les tests du module : `pytest backend/tests/fichier.py -v`
5. ✅ Vérifier le coverage : `pytest --cov=backend/module backend/tests/test_module.py`
6. ✅ Exécuter la suite complète pour détecter les régressions
7. ✅ Commit avec message descriptif
8. ✅ Créer une PR pour review

---

## NOTES IMPORTANTES

- **PostgreSQL** : Tous les tests utilisent PostgreSQL (pas SQLite) [[memory:10130800]]
- **Environnement de test** : Les tests utilisent des savepoints pour isolation
- **Fixtures** : Vérifier `backend/tests/conftest.py` pour les fixtures disponibles
- **Configuration** : Les tests utilisent `FLASK_ENV=testing`

---

**Dernière mise à jour** : 2025-11-20

