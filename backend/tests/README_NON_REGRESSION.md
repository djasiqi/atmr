# 🛡️ Tests de Non-Régression - ATMR Backend

Ce document liste et documente tous les tests de non-régression du projet ATMR. Ces tests sont critiques pour prévenir les régressions et garantir la stabilité du système.

## 📋 Vue d'ensemble

Les tests de non-régression sont des tests qui vérifient qu'un comportement spécifique, une fois corrigé, ne se reproduit plus. Ils sont marqués avec `✅ Test de non-régression` dans leur docstring.

## 🎯 Objectif

Les tests de non-régression servent à :

- ✅ Prévenir la réapparition de bugs connus
- ✅ Garantir la stabilité des correctifs appliqués
- ✅ Documenter les comportements attendus du système
- ✅ Faciliter le debugging en cas de régression

## 📚 Liste des Tests de Non-Régression

### 1. Gestion des Sessions et Transactions DB

#### `test_company_persisted_before_dispatch`

**Fichier** : `backend/tests/e2e/test_dispatch_e2e.py:474`

**Objectif** : Vérifier que la Company est bien persistée avant dispatch.

**Problème résolu** : Les fixtures `company` doivent être commitées avant `engine.run()` car cette fonction fait un rollback défensif qui peut expirer les objets non commités.

**Vérifications** :

- La Company existe en DB après création
- `engine.run()` peut trouver la Company et créer un DispatchRun
- Le DispatchRun est correctement lié à la Company

**Impact** : Critique - Sans ce test, les dispatches peuvent échouer avec des erreurs FK.

---

#### `test_fixtures_isolation_and_rollback_defensive`

**Fichier** : `backend/tests/e2e/test_dispatch_e2e.py:500`

**Objectif** : Vérifier l'isolation des fixtures et le rollback défensif.

**Problème résolu** : Les fixtures doivent être isolées entre les tests, et le rollback défensif de `engine.run()` ne doit pas affecter les objets commités.

**Vérifications** :

- Les fixtures sont bien isolées (savepoints)
- Le rollback défensif n'affecte pas les objets commités
- Les objets restent visibles après `engine.run()`

**Impact** : Critique - Garantit l'isolation entre les tests et la stabilité des fixtures.

---

#### `test_rollback_restores_original_values`

**Fichier** : `backend/tests/e2e/test_dispatch_e2e.py:452`

**Objectif** : Vérifier que le rollback restaure bien les valeurs originales.

**Problème résolu** : Après un rollback, les objets SQLAlchemy doivent être rechargés depuis la DB pour garantir que les valeurs sont bien restaurées.

**Vérifications** :

- Les modifications non commitées sont bien annulées après rollback
- Les objets rechargés depuis la DB ont les bonnes valeurs
- Le rollback restaure l'état initial

**Impact** : Critique - Garantit l'intégrité des données après rollback.

---

#### Tests de robustesse des rollbacks (`test_rollback_robustness.py`)

**Fichier** : `backend/tests/e2e/test_rollback_robustness.py`

**Objectif** : Vérifier systématiquement que les rollbacks restaurent correctement les valeurs dans différents scénarios.

**Problème résolu** : Garantir que les rollbacks fonctionnent correctement dans tous les cas (champ unique, plusieurs champs, plusieurs objets, après flush, après commit partiel, après rollback défensif).

**Tests inclus** :

- `test_rollback_restores_single_field` - Vérifie qu'un champ unique est restauré
- `test_rollback_restores_multiple_fields` - Vérifie que plusieurs champs sont restaurés
- `test_rollback_restores_multiple_objects` - Vérifie que plusieurs objets sont restaurés
- `test_rollback_restores_after_flush` - Vérifie après flush (ID assigné mais non commité)
- `test_rollback_restores_after_partial_commit` - Vérifie après commit partiel
- `test_rollback_restores_after_engine_run_rollback_defensive` - Vérifie après rollback défensif
- `test_rollback_restores_assignment_after_dispatch_failure` - Vérifie après échec de dispatch

**Vérifications** :

- Utilisation des helpers `verify_rollback_restores_values()` et `capture_original_values()`
- Vérification systématique de la restauration des valeurs
- Gestion de l'expiration des objets après rollback
- Rechargement depuis la DB avec stratégies configurables

**Impact** : Critique - Garantit la robustesse des rollbacks dans tous les scénarios.

---

#### `test_apply_assignments_finds_bookings`

**Fichier** : `backend/tests/e2e/test_dispatch_e2e.py:420`

**Objectif** : Vérifier que `apply_assignments()` trouve bien les bookings après commit.

**Problème résolu** : Les bookings doivent être commités avant d'être utilisés par `apply_assignments()`, sinon ils ne sont pas trouvés lors de la requête DB.

**Vérifications** :

- Les bookings sont trouvés après commit
- `apply_assignments()` peut accéder aux bookings
- Les assignments sont correctement créés

**Impact** : Critique - Sans ce test, les assignments peuvent échouer silencieusement.

---

### 2. Gestion des Erreurs et Exceptions

#### `test_company_not_found_raises_exception`

**Fichier** : `backend/tests/e2e/test_dispatch_e2e.py:541`

**Objectif** : Vérifier que `CompanyNotFoundError` est levée si demandé.

**Problème résolu** : Permettre une gestion d'erreur explicite via exception au lieu d'un retour structuré.

**Vérifications** :

- Comportement par défaut : retourne un résultat structuré avec `reason="company_not_found"`
- Comportement avec `raise_on_company_not_found=True` : lève `CompanyNotFoundError`
- L'exception contient les bonnes informations (company_id, caller, etc.)

**Impact** : Important - Permet une gestion d'erreur explicite dans les cas internes.

---

### 3. Métriques et Observabilité

#### `test_osrm_metrics_initialized`

**Fichier** : `backend/tests/e2e/test_dispatch_metrics_e2e.py:253`

**Objectif** : Vérifier que les métriques OSRM sont initialisées même sans appels.

**Problème résolu** : Les métriques Prometheus doivent être déclarées (HELP/TYPE) même si elles n'ont jamais été incrémentées, pour apparaître dans l'endpoint `/metrics`.

**Vérifications** :

- Les métriques sont déclarées (HELP/TYPE présents)
- Les métriques sont initialisées avec 0.0
- Les métriques apparaissent dans l'endpoint `/metrics` même sans appels OSRM

**Impact** : Important - Garantit l'observabilité complète du système.

---

### 4. Sécurité et Middleware

#### `test_no_redirects_in_testing_mode`

**Fichier** : `backend/tests/e2e/test_disaster_scenarios.py:663`

**Objectif** : Vérifier qu'aucune redirection 302 n'est générée en mode testing.

**Problème résolu** : Talisman middleware ne doit pas forcer HTTPS en mode testing, ce qui causait des redirections 302 inattendues dans les tests.

**Vérifications** :

- Aucune redirection 302 pour les routes API
- Les codes HTTP sont corrects (200, 400, 401, etc.)
- Talisman est désactivé en mode testing

**Impact** : Critique - Les tests E2E doivent pouvoir vérifier les codes HTTP directement.

---

#### `test_no_redirects_in_auth_endpoints`

**Fichier** : `backend/tests/e2e/test_schema_validation.py:19`

**Objectif** : Vérifier l'absence de redirections dans les endpoints auth.

**Problème résolu** : Les endpoints d'authentification ne doivent pas générer de redirections 302 inattendues.

**Vérifications** :

- Les endpoints `/api/v1/auth/login` et `/api/v1/auth/register` ne redirigent pas
- Les codes HTTP sont corrects (200, 400, 401, etc.)
- Les réponses JSON sont valides

**Impact** : Critique - Les tests d'authentification doivent pouvoir vérifier les codes HTTP.

---

## 🔍 Scénarios Critiques à Surveiller

### Scénarios déjà couverts ✅

- ✅ Persistance des fixtures avant `engine.run()`
- ✅ Isolation des fixtures entre les tests
- ✅ Restauration des valeurs après rollback
- ✅ Visibilité des objets après commit
- ✅ Initialisation des métriques Prometheus
- ✅ Absence de redirections 302 en mode testing
- ✅ Gestion des exceptions personnalisées

### Scénarios potentiels à ajouter (optionnel) ⚠️

Les scénarios suivants sont **optionnels** car :

1. Ils sont déjà partiellement testés dans d'autres types de tests (unitaires, intégration, edge cases)
2. Ils ne sont pas des régressions connues mais des cas limites
3. Leur implémentation nécessiterait des tests complexes ou des outils spécialisés

#### ⚠️ **Gestion des timeouts**

**Statut** : Partiellement testé dans les tests unitaires et edge cases

**Tests existants** :

- `test_osrm_timeout_raises_exception` (`backend/tests/test_osrm_client.py:81`)
- `test_osrm_service_timeout` (`backend/tests/rl/test_osrm_fallback_edge_cases.py:37`)

**Pourquoi optionnel** :

- Les timeouts sont déjà testés dans les tests unitaires
- Les tests de non-régression se concentrent sur les bugs connus, pas les cas limites
- Les timeouts sont gérés par les bibliothèques externes (requests, etc.)

**Recommandation** : Maintenir les tests unitaires existants, ajouter un test de non-régression uniquement si un bug spécifique de timeout est identifié.

---

#### ⚠️ **Gestion de la mémoire**

**Statut** : Non testé (nécessiterait des outils spécialisés)

**Pourquoi optionnel** :

- Les fuites mémoire sont difficiles à détecter dans des tests automatisés
- Nécessiterait des outils spécialisés (memory_profiler, tracemalloc)
- Les tests de non-régression se concentrent sur les bugs fonctionnels, pas les problèmes de performance
- Les fuites mémoire sont généralement détectées en production via monitoring

**Recommandation** : Utiliser le monitoring en production plutôt que des tests automatisés. Ajouter un test de non-régression uniquement si une fuite mémoire spécifique est identifiée.

---

#### ⚠️ **Gestion des connexions DB**

**Statut** : Partiellement testé via les fixtures et les tests d'isolation

**Tests existants** :

- Les fixtures `db` garantissent l'isolation via savepoints
- `test_fixtures_isolation_and_rollback_defensive` vérifie l'isolation

**Pourquoi optionnel** :

- Les connexions DB sont gérées par SQLAlchemy et les fixtures
- L'isolation est déjà testée via les tests de non-régression existants
- Les connexions sont automatiquement fermées par les fixtures (via `db.session.remove()`)

**Recommandation** : Maintenir les tests d'isolation existants. Ajouter un test de non-régression uniquement si un problème spécifique de connexion est identifié.

---

#### ⚠️ **Gestion des erreurs réseau**

**Statut** : Partiellement testé dans les tests d'intégration et edge cases

**Tests existants** :

- `test_osrm_fallback` (`backend/tests/integration/test_osrm_fallback.py`)
- `test_rl_task_network_failure` (`backend/tests/rl/test_rl_celery_edge_cases.py:123`)
- `test_osrm_service_rate_limit` (`backend/tests/rl/test_osrm_fallback_edge_cases.py:121`)

**Pourquoi optionnel** :

- Les erreurs réseau sont déjà testées dans les tests d'intégration
- Les tests de non-régression se concentrent sur les bugs connus, pas les cas limites
- Les erreurs réseau sont gérées par les mécanismes de fallback (déjà testés)

**Recommandation** : Maintenir les tests d'intégration existants. Ajouter un test de non-régression uniquement si un bug spécifique de gestion d'erreur réseau est identifié.

---

#### ⚠️ **Gestion des erreurs de validation**

**Statut** : Testé dans les tests de validation et schema validation

**Tests existants** :

- `test_schema_validation.py` - Tests complets de validation
- `test_validation_schemas.py` - Tests de validation des schémas
- `test_input_validation.py` - Tests de validation des entrées

**Pourquoi optionnel** :

- Les erreurs de validation sont déjà largement testées dans les tests de validation dédiés
- Les tests de non-régression se concentrent sur les bugs connus, pas les cas de validation standards
- Les erreurs de validation sont gérées par Marshmallow (bibliothèque externe testée)

**Recommandation** : Maintenir les tests de validation existants. Ajouter un test de non-régression uniquement si un bug spécifique de validation est identifié.

---

## 📊 Résumé des Scénarios Optionnels

| Scénario                          | Statut              | Tests Existants    | Priorité   | Action Recommandée          |
| --------------------------------- | ------------------- | ------------------ | ---------- | --------------------------- |
| Gestion des timeouts              | Partiellement testé | ✅ Oui             | Basse      | Maintenir tests unitaires   |
| Gestion de la mémoire             | Non testé           | ❌ Non             | Très basse | Monitoring production       |
| Gestion des connexions DB         | Partiellement testé | ✅ Oui (isolation) | Basse      | Maintenir tests isolation   |
| Gestion des erreurs réseau        | Partiellement testé | ✅ Oui             | Basse      | Maintenir tests intégration |
| Gestion des erreurs de validation | Testé               | ✅ Oui             | Basse      | Maintenir tests validation  |

**Conclusion** : Les scénarios optionnels sont soit déjà testés dans d'autres types de tests, soit non critiques pour des tests de non-régression. Aucun test de non-régression supplémentaire n'est nécessaire pour l'instant.

## 📝 Bonnes Pratiques

### 1. Nommage des Tests

Les tests de non-régression doivent :

- Commencer par `test_` (convention pytest)
- Avoir une docstring avec `✅ Test de non-régression :`
- Décrire clairement le problème résolu

### 2. Structure des Tests

```python
def test_example_non_regression(self, db, company):
    """✅ Test de non-régression : Vérifier que [comportement spécifique].

    Problème résolu : [Description du problème]

    Vérifications :
    - [Vérification 1]
    - [Vérification 2]
    """
    # Arrange
    # Act
    # Assert
    pass
```

### 3. Maintenance

- ✅ Exécuter les tests de non-régression à chaque commit
- ✅ Ne pas supprimer un test de non-régression sans justification
- ✅ Mettre à jour ce document si un nouveau test est ajouté
- ✅ Documenter les raisons si un test est supprimé

## 🔗 Références

- [Guide des Fixtures et Isolation](./README_FIXTURES.md)
- [Documentation Pytest](https://docs.pytest.org/)
- [Audit CI/Pytest 2025](../../docs/audit-ci-pytest-2025.md)

## 📊 Statistiques

- **Total de tests de non-régression** : 7
- **Couverture** : Sessions DB, Transactions, Métriques, Sécurité, Exceptions
- **Dernière mise à jour** : 2025-01-XX

---

**Note** : Ce document doit être mis à jour à chaque ajout/suppression de test de non-régression.
