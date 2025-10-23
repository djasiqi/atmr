# 🔗 Intégration de l'Optimiseur RL dans le Dispatch

**Date** : 21 octobre 2025  
**Statut** : ⏳ **PRÊT** (en attente de la fin de l'entraînement)

---

## 📦 Fichiers Créés

✅ **`backend/services/unified_dispatch/rl_optimizer.py`**  
→ Classe `RLDispatchOptimizer` qui améliore les assignations heuristiques

---

## 🔧 Intégration dans `engine.py`

### Étape 1 : Import de l'Optimiseur

Ajouter en haut de `backend/services/unified_dispatch/engine.py` :

```python
from services.unified_dispatch.rl_optimizer import RLDispatchOptimizer
```

### Étape 2 : Appliquer l'Optimisation

Dans la fonction `run()`, **après l'heuristique**, ajouter :

```python
# 🧠 Optimisation RL (si activée et modèle disponible)
if mode == "auto" and len(final_assignments) > 0:
    try:
        logger.info("[Engine] 🧠 Tentative d'optimisation RL des assignations...")

        optimizer = RLDispatchOptimizer(
            model_path="data/rl/models/dispatch_optimized_v1.pth",
            max_swaps=10,  # Max 10 réassignations
            min_improvement=0.5,  # Amélioration minimale de 0.5 course
        )

        if optimizer.is_available():
            # Convertir assignments en format optimisable
            initial = [
                {
                    "booking_id": a.booking_id,
                    "driver_id": a.driver_id,
                }
                for a in final_assignments
            ]

            # Optimiser
            optimized = optimizer.optimize_assignments(
                initial_assignments=initial,
                bookings=problem["bookings"],
                drivers=regs,
            )

            # Appliquer les changements
            for i, a in enumerate(final_assignments):
                new_driver_id = optimized[i]["driver_id"]
                if a.driver_id != new_driver_id:
                    logger.info(
                        "[Engine] RL swap: Booking %d → Driver %d (was %d)",
                        a.booking_id,
                        new_driver_id,
                        a.driver_id,
                    )
                    a.driver_id = new_driver_id

            logger.info("[Engine] ✅ Optimisation RL terminée")
        else:
            logger.info("[Engine] ⏳ Optimiseur RL non disponible (modèle en cours d'entraînement)")

    except Exception as e:
        logger.warning("[Engine] ⚠️ Optimisation RL échouée: %s", e)
        # Continuer avec l'heuristique seule
```

### Position d'Insertion

Insérer ce code **ligne ~490** dans `engine.py`, juste après :

```python
# ... Heuristique P1 a assigné toutes les courses ...
logger.info("[Engine] Heuristic P1: %d assignés, %d restants",
            len(h_res.assignments), len(h_res.unassigned_booking_ids))

# 🆕 INSÉRER L'OPTIMISATION RL ICI

# ⚠️ Vérification d'équité : TEMPORAIREMENT DÉSACTIVÉE
if False:  # Désactivé temporairement - voir commentaires ci-dessus
```

---

## 🎯 Comportement de l'Optimiseur

### Mode de Fonctionnement

1. **Détection Automatique** :

   - Vérifie si le modèle existe (`data/rl/models/dispatch_optimized_v1.pth`)
   - Si absent → Skip (pas d'erreur, continue avec heuristique)
   - Si présent → Active l'optimisation

2. **Optimisation** :

   - Prend les assignations de l'heuristique
   - Calcule l'écart initial (ex: 5-3-2 → gap=3)
   - Suggère jusqu'à 10 réassignations
   - Valide chaque swap (amélioration ≥0.5 ?)
   - Applique uniquement les swaps bénéfiques

3. **Critères de Succès** :
   - **Objectif Principal** : Réduire l'écart de charge
   - **Contraintes** : Respecter les time windows
   - **Arrêt** : Gap ≤1 OU 10 swaps atteints

### Logs Produits

```
[Engine] 🧠 Tentative d'optimisation RL des assignations...
[RLOptimizer] ✅ Modèle chargé: data/rl/models/dispatch_optimized_v1.pth
[RLOptimizer] 🧠 Début optimisation: 10 assignments, 3 drivers
[RLOptimizer] Écart initial: 3 courses
[RLOptimizer] ✅ Swap 1/10 accepté: Booking 169 → Driver 3 (gap 3 → 2, Δ=1.0)
[RLOptimizer] ✅ Swap 2/10 accepté: Booking 156 → Driver 4 (gap 2 → 1, Δ=1.0)
[RLOptimizer] 🎯 Optimal atteint (gap=1), arrêt
[RLOptimizer] 🎉 Optimisation terminée: gap 3 → 1 (10 swaps, 2 améliorations)
[Engine] ✅ Optimisation RL terminée
```

---

## ⚙️ Configuration

### Paramètres de l'Optimiseur

| Paramètre         | Valeur                                     | Description                                 |
| ----------------- | ------------------------------------------ | ------------------------------------------- |
| `model_path`      | `data/rl/models/dispatch_optimized_v1.pth` | Chemin du modèle entraîné                   |
| `max_swaps`       | 10                                         | Nombre max de réassignations à tenter       |
| `min_improvement` | 0.5                                        | Amélioration minimale pour accepter un swap |

### Feature Flag (Optionnel)

Pour activer/désactiver facilement l'optimiseur RL, ajouter dans `company.autonomous_config` :

```json
{
  "features": {
    "enable_rl_optimization": true
  }
}
```

Puis dans `engine.py` :

```python
if mode == "auto" and getattr(s.features, "enable_rl_optimization", True):
    # ... optimisation RL ...
```

---

## 📊 Résultats Attendus

### Avant (Heuristique Seule)

```
Giuseppe : 5 courses  ❌
Dris     : 3 courses
Yannis   : 2 courses
ÉCART    : 3
```

### Après (Heuristique + RL Optimizer)

```
Giuseppe : 4 courses  ✅
Dris     : 3 courses  ✅
Yannis   : 3 courses  ✅
ÉCART    : 1
```

**Amélioration** : **Écart réduit de 66%** (3 → 1) 🎉

---

## 🧪 Tests de Validation

### Test 1 : Modèle Absent

**Setup** : Pas de fichier `.pth`  
**Comportement attendu** :

- Log : "Optimiseur RL non disponible"
- Retour assignations heuristiques intactes
- ✅ Pas d'erreur, dispatch fonctionne normalement

### Test 2 : Modèle Présent, Déjà Optimal

**Setup** : Gap initial = 1  
**Comportement attendu** :

- Log : "Déjà optimal (gap=1), pas d'optimisation"
- Retour assignations inchangées
- ✅ Économie de calcul

### Test 3 : Modèle Présent, Optimisation Nécessaire

**Setup** : Gap initial = 3  
**Comportement attendu** :

- Log : "Optimisation terminée: gap 3 → 1"
- Assignations modifiées
- ✅ Équité améliorée

---

## 🚀 Activation en Production

### Étape 1 : Attendre la Fin de l'Entraînement

```bash
# Vérifier la progression
docker exec atmr-api-1 python backend/scripts/monitor_rl_training.py

# Attendre "✅ ENTRAÎNEMENT TERMINÉ !"
```

### Étape 2 : Vérifier le Modèle

```bash
# Vérifier que le fichier existe
docker exec atmr-api-1 ls -lh data/rl/models/dispatch_optimized_v1.pth

# Taille attendue : ~1-5 MB
```

### Étape 3 : Intégrer dans `engine.py`

Ajouter le code d'intégration (voir ci-dessus)

### Étape 4 : Redémarrer les Services

```bash
docker restart atmr-api-1
docker restart atmr-celery-worker-1
```

### Étape 5 : Tester sur un Dispatch

1. Aller dans l'UI : Dispatch Semi-Auto
2. Sélectionner date : 23.10.2025
3. Cliquer "Lancer le Dispatch"
4. Vérifier les logs :
   ```bash
   docker logs atmr-celery-worker-1 --tail 100 | grep "RLOptimizer"
   ```
5. Comparer la répartition avant/après

---

## 📈 Monitoring

### Métriques à Suivre

1. **Écart de Charge** :

   - Avant : Heuristique seule
   - Après : Heuristique + RL
   - Objectif : Réduction ≥50%

2. **Temps d'Exécution** :

   - Heuristique : ~5s
   - RL : +2-3s supplémentaires
   - Total acceptable : <10s

3. **Taux de Succès** :
   - % de dispatches avec gap ≤1
   - Objectif : ≥80%

### Logs à Monitorer

```bash
# Succès d'optimisation
docker logs atmr-celery-worker-1 | grep "Optimisation terminée"

# Swaps effectués
docker logs atmr-celery-worker-1 | grep "Swap.*accepté"

# Erreurs éventuelles
docker logs atmr-celery-worker-1 | grep "RLOptimizer.*❌"
```

---

## 🔄 Réentraînement

L'optimiseur peut être amélioré continuellement :

1. **Collecter plus de données** :

   - Exporter dispatches de toute la semaine
   - Script : `rl_export_historical_data.py`

2. **Réentraîner** :

   - Lancer `rl_train_offline.py` avec nouvelles données
   - Le modèle sera sauvegardé dans le même fichier

3. **Activer automatiquement** :
   - Pas besoin de modifier le code
   - L'optimiseur rechargera le nouveau modèle

---

## ⚠️ Dépannage

### Problème : "Modèle non trouvé"

**Cause** : Fichier `.pth` absent  
**Solution** : Attendre la fin de l'entraînement (2-3h)

### Problème : "Optimisation RL échouée"

**Cause** : Erreur de chargement du modèle  
**Solution** :

1. Vérifier les logs détaillés
2. Vérifier compatibilité PyTorch
3. Réentraîner si nécessaire

### Problème : "Pas d'amélioration"

**Cause** : Modèle sous-entraîné ou données insuffisantes  
**Solution** :

1. Collecter plus de dispatches historiques
2. Réentraîner avec 10,000 épisodes
3. Ajuster `min_improvement` (0.3 au lieu de 0.5)

---

## 📝 Notes Importantes

- ✅ **Pas d'impact si modèle absent** : Le dispatch fonctionne normalement
- ✅ **Fallback automatique** : En cas d'erreur, retour à l'heuristique
- ✅ **Validation systématique** : Chaque swap est vérifié avant application
- ✅ **Logs détaillés** : Traçabilité complète des décisions

---

**Dernière mise à jour** : 21 octobre 2025, 23:30  
**Statut** : Prêt pour intégration (dès que l'entraînement est terminé)
