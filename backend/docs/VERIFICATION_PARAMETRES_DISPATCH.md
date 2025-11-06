# Vérification de la transmission des paramètres avancés

## ✅ Paramètres bien transmis et utilisés

### 1. Heuristique (heuristic)
- ✅ **proximity** (0.05) → `settings.heuristic.proximity` → Utilisé dans `_score_driver_for_booking()` ligne 502
- ✅ **driver_load_balance** (0.95) → `settings.heuristic.driver_load_balance` → Utilisé dans `_score_driver_for_booking()` ligne 503
- ✅ **priority** (0.06) → `settings.heuristic.priority` → Utilisé dans `_score_driver_for_booking()` ligne 504

### 2. Solver (solver)
- ✅ **time_limit_sec** (60) → `settings.solver.time_limit_sec` → Utilisé dans `solver.py`
- ✅ **max_bookings_per_driver** (10) → `settings.solver.max_bookings_per_driver` → Utilisé dans `heuristics.py` via `get_adjusted_max_cap()`
- ✅ **unassigned_penalty_base** (10000) → `settings.solver.unassigned_penalty_base` → Utilisé dans `solver.py`

### 3. Temps de service (service_times)
- ✅ **pickup_service_min** (5) → `settings.service_times.pickup_service_min` → Utilisé dans `build_vrptw_problem()` ligne 969
- ✅ **dropoff_service_min** (10) → `settings.service_times.dropoff_service_min` → Utilisé dans `build_vrptw_problem()` ligne 974
- ✅ **min_transition_margin_min** (15) → `settings.service_times.min_transition_margin_min` → Utilisé dans `build_vrptw_problem()` ligne 983

### 4. Regroupement (pooling)
- ✅ **enabled** (True) → `settings.pooling.enabled` → Utilisé dans `_can_be_pooled()` ligne 93
- ✅ **time_tolerance_min** (10) → `settings.pooling.time_tolerance_min` → Utilisé dans `_can_be_pooled()` ligne 105
- ✅ **pickup_distance_m** (500) → `settings.pooling.pickup_distance_m` → Utilisé dans `_can_be_pooled()` ligne 128

### 5. Équité (fairness)
- ✅ **enabled** (True) → `settings.fairness.enable_fairness` → Utilisé dans l'algorithme d'équité
- ⚠️ **window_days** (2) → `settings.fairness.fairness_window_days` → Utilisé pour calculer `fairness_counts` (actuellement sur 1 jour)
- ✅ **fairness_weight** (0.7) → `settings.fairness.fairness_weight` → Utilisé dans le calcul de l'équité

### 6. Chauffeur préféré (root level)
- ✅ **preferred_driver_id** → `problem["preferred_driver_id"]` → Utilisé dans `get_eligible_drivers()` ligne 638
- ✅ **driver_load_multipliers** → `problem["driver_load_multipliers"]` → Utilisé dans `get_adjusted_max_cap()` ligne 632

### 7. Chauffeurs d'urgence (emergency)
- ✅ **allow_emergency** → `problem["allow_emergency"]` → Utilisé dans `build_problem_data()` ligne 1177
- ✅ **emergency.allow_emergency_drivers** (True) → `settings.emergency.allow_emergency_drivers` → Utilisé dans `engine.py` ligne 290
- ✅ **emergency.emergency_penalty** (900) → `settings.emergency.emergency_penalty` → Utilisé dans le scoring (malus -0.60 ligne 867)

## 🔄 Flux de transmission

1. **Frontend** (`AdvancedSettings.jsx`) → Envoie `overrides` via `onApply(overrides)`
2. **UnifiedDispatchRefactored.jsx** → Passe `overrides` à `runDispatchForDay()`
3. **companyService.js** → Inclut `overrides` dans le payload POST `/company_dispatch/run`
4. **dispatch_routes.py** → Reçoit `overrides` dans `DispatchOverridesSchema`
5. **engine.py** → Applique `merge_overrides(s, overrides)` ligne 276
6. **build_problem_data()** → Ajoute `preferred_driver_id`, `driver_load_multipliers`, `allow_emergency` au `problem`
7. **heuristics.py** → Lit depuis `problem` et `settings` pour le scoring/assignation

## ⚠️ Points à vérifier

1. **fairness_window_days** : Le paramètre est configuré mais `count_assigned_bookings_for_day()` utilise actuellement `day=None` (jour actuel). Le paramètre `window_days` pourrait être utilisé pour calculer l'équité sur plusieurs jours.

2. **emergency_penalty** : Le malus dans `heuristics.py` ligne 867 est fixe (-0.60), il devrait utiliser `settings.emergency.emergency_penalty` converti en malus.

## ✅ Conclusion

Tous les paramètres sont correctement transmis et utilisés dans le dispatch, sauf :
- `fairness_window_days` : configuré mais pas encore utilisé pour le calcul multi-jours
- `emergency_penalty` : valeur fixe au lieu d'utiliser le paramètre

