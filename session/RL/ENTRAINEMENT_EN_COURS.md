# 🧠 Entraînement RL en Cours

**Date** : 21 octobre 2025, 23h20  
**Statut** : ✅ **EN COURS** (2% complété)

---

## 📊 État Actuel

```
Progression : 100/5000 épisodes (2.0%)
Écart moyen : 4.96 courses
Durée estimée : ~2-3 heures
```

**Processus en arrière-plan** ✅  
L'entraînement tourne dans le conteneur Docker `atmr-api-1`

---

## 📈 Suivre la Progression

### Méthode 1 : Script de Monitoring

```bash
docker exec atmr-api-1 python backend/scripts/monitor_rl_training.py
```

### Méthode 2 : Logs en Temps Réel

```bash
docker exec atmr-api-1 tail -f data/rl/training_output.log
```

### Méthode 3 : Dernières Lignes

```bash
docker exec atmr-api-1 tail -20 data/rl/training_output.log
```

---

## 🎯 Objectif de l'Entraînement

L'agent DQN apprend à **minimiser l'écart de charge** entre chauffeurs :

### État Actuel (Heuristique)

```
Giuseppe : 5 courses  ❌
Dris     : 3 courses
Yannis   : 2 courses
ÉCART    : 3
```

### Objectif (Après RL)

```
Giuseppe : 3-4 courses  ✅
Dris     : 3-4 courses  ✅
Yannis   : 3-4 courses  ✅
ÉCART    : 0-1
```

---

## ⚙️ Configuration de l'Entraînement

| Paramètre            | Valeur     | Description                                |
| -------------------- | ---------- | ------------------------------------------ |
| **Épisodes**         | 5000       | Nombre d'itérations d'apprentissage        |
| **État (dimension)** | 94         | Positions chauffeurs + bookings + contexte |
| **Actions**          | 61         | Assigner booking[i] → driver[j]            |
| **Learning Rate**    | 0.0001     | Taux d'apprentissage                       |
| **Batch Size**       | 64         | Taille des batchs d'entraînement           |
| **Buffer Size**      | 10,000     | Mémoire des expériences                    |
| **Epsilon**          | 0.5 → 0.01 | Exploration → Exploitation                 |

---

## 📦 Réseau de Neurones

```
Q-Network (DQN)
├── Input Layer    : 94 neurones (état)
├── Hidden Layer 1 : 256 neurones + ReLU
├── Hidden Layer 2 : 256 neurones + ReLU
└── Output Layer   : 61 neurones (Q-values)

Total : 220,733 paramètres entraînables
```

---

## 🔄 Sauvegardes Automatiques

Le modèle est **sauvegardé tous les 100 épisodes** si amélioration détectée :

```
📂 data/rl/models/dispatch_optimized_v1.pth
```

Critères de sauvegarde :

1. **Priorité** : Écart de charge réduit (gap < meilleur_précédent)
2. **Secondaire** : Récompense améliorée (à gap égal)

---

## ⏱️ Timeline Estimée

| Temps    | Épisodes | Progression |
| -------- | -------- | ----------- |
| 0 min    | 0        | Démarrage   |
| 15 min   | 500      | 10%         |
| 30 min   | 1000     | 20%         |
| 1h       | 2000     | 40%         |
| 1h30     | 3000     | 60%         |
| 2h       | 4000     | 80%         |
| **2h30** | **5000** | **100% ✅** |

---

## 📊 Métriques Suivies

### 1. Écart de Charge (Load Gap)

- **Actuel** : 4.96 courses
- **Objectif** : ≤1 course
- **Poids dans la récompense** : Critique (-20 × gap²)

### 2. Récompense Totale

- **Actuelle** : -2369.80
- **Objectif** : > -500
- **Composition** :
  - Équité : -20 × (écart)²
  - Bonus écart ≤1 : +100
  - Distance : -0.5 × km

### 3. Distance Totale

- **Actuelle** : 0.0 km (données manquantes)
- **Objectif** : Minimiser
- **Priorité** : Secondaire (après équité)

---

## 🚀 Prochaines Étapes

### Pendant l'Entraînement (maintenant)

- [x] Export des données historiques
- [x] Lancement de l'entraînement
- [ ] Préparation de l'intégrateur RL
- [ ] Tests sur données de validation

### Après l'Entraînement (dans ~2-3h)

1. **Évaluation du modèle** :

   - Charger `dispatch_optimized_v1.pth`
   - Tester sur dispatch du 22 octobre
   - Comparer : heuristique vs RL

2. **Intégration dans le dispatch** :

   - Créer `RLDispatchOptimizer`
   - Modifier `engine.py`
   - Activer en mode "auto"

3. **Validation** :
   - A/B testing (avec/sans RL)
   - Monitoring des métriques
   - Ajustements si nécessaire

---

## 🛑 Arrêter l'Entraînement

Si nécessaire (problème, erreur, etc.) :

```bash
# Trouver le PID du processus Python
docker exec atmr-api-1 ps aux | grep "rl_train_offline"

# Tuer le processus
docker exec atmr-api-1 kill <PID>

# Ou redémarrer le conteneur
docker restart atmr-api-1
```

---

## 📝 Notes Importantes

### Limitations Actuelles

- **1 seul dispatch historique** exporté (22 oct.)
  - Idéal : 50-100 dispatches
  - Impact : Généralisation limitée
- **Coordonnées GPS manquantes** pour certaines courses
  - Distance = 0 km dans les métriques
  - L'agent optimise donc principalement l'équité

### Améliorations Futures

1. **Exporter plus de dispatches** (semaine entière)
2. **Ajouter contexte temporel** (heure, jour de la semaine)
3. **Intégrer données OSRM** (temps réel de trajet)
4. **Multi-objectif** (équité + distance + satisfaction client)

---

## 🎓 Ce Que l'Agent Apprend

L'agent DQN découvre automatiquement :

1. **Patterns de charge** :

   - Éviter d'assigner trop de courses au même chauffeur
   - Équilibrer la charge dès le début du dispatch

2. **Contraintes temporelles** :

   - Respecter les fenêtres de temps
   - Prioriser les urgences

3. **Stratégies optimales** :
   - Quand attendre (action 0)
   - Quel chauffeur choisir pour chaque course
   - Comment minimiser l'écart final

---

## 📞 Contact

En cas de question ou problème :

- Vérifier les logs : `data/rl/training_output.log`
- Monitoring : `monitor_rl_training.py`
- Documentation : `session/RL/PLAN_ENTRAINEMENT_DISPATCH_OPTIMAL.md`

---

**Dernière mise à jour** : 21 octobre 2025, 23:20  
**Prochaine vérification recommandée** : Dans 30 minutes (≈20% complété)
