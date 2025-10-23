# 🚀 Amélioration du Modèle RL avec Données Excel

**Date** : 21 octobre 2025  
**Statut** : 🔄 **EN COURS**

---

## 🎯 OBJECTIF

Améliorer le modèle RL en utilisant **1 année de données historiques** issues d'un fichier Excel pour atteindre **gap ≤1 systématiquement**.

---

## 📊 DONNÉES DISPONIBLES

### Fichier Source : `transport.xlsx`

```
📑 Feuille : Feuil1
📦 211 courses (octobre 2025)
📋 6 colonnes
```

### Structure des Données

| Colonne                 | Contenu             | Exemple                                         |
| ----------------------- | ------------------- | ----------------------------------------------- |
| `Nom/Prénom`            | Client              | "REYTAN Catherine"                              |
| `Date et Heure prévues` | Date + heures       | "01.10.2025" "09:15" "16:00"                    |
| `Course`                | Type                | "A/R" (Aller-Retour)                            |
| `Adresse de départ`     | Texte complet       | "Chemin des Ramiers 9, 1245 Collonge-Bellerive" |
| `Adresse d'arrivée`     | Texte complet       | "Route d'Hermance 347, 1247 Anières"            |
| `CFT`                   | Initiales chauffeur | "Y.L", "D.D", "G.B"                             |

### Mapping Chauffeurs Confirmé

| Initiales | Nom Complet     | ID  |
| --------- | --------------- | --- |
| **Y.L**   | Yannis Labrot   | 2   |
| **D.D**   | Dris Daoudi     | 4   |
| **G.B**   | Giuseppe Bekasy | 3   |
| **K.A**   | Khalid Alaoui   | 1   |

---

## 🔄 PROCESSUS DE CONVERSION

### Étape 1 : Lecture Excel ✅

```python
df = pd.read_excel("transport.xlsx")
# 211 courses chargées
```

### Étape 2 : Géocodage des Adresses 🔄

**En cours** : ~7-10 minutes

- **211 courses** × 2 adresses = **422 adresses à géocoder**
- **API** : Nominatim (OpenStreetMap, gratuit)
- **Limite** : 1 requête/seconde
- **Cache** : `data/rl/geocode_cache.json` (réutilisable)

```
Adresse départ : "Chemin des Ramiers 9, 1245 Collonge-Bellerive"
       ↓ Géocodage
Coordonnées    : (46.2531, 6.1842)
```

### Étape 3 : Calcul des Distances ⏳

```python
distance_km = haversine_distance(
    (pickup_lat, pickup_lon),
    (dropoff_lat, dropoff_lon)
)
```

### Étape 4 : Formatage RL ⏳

```json
{
  "dispatches": [
    {
      "date": "2025-10-01",
      "num_bookings": 12,
      "num_drivers": 3,
      "driver_loads": {"2": 4, "3": 5, "4": 3},
      "load_gap": 2,
      "bookings": [...]
    },
    ...
  ]
}
```

---

## 📈 AMÉLIORATION ATTENDUE

### Comparaison

| Métrique                       | Modèle Actuel (v1) | Modèle Futur (v2)  | Amélioration |
| ------------------------------ | ------------------ | ------------------ | ------------ |
| **Données d'entraînement**     | 1 dispatch         | **~30 dispatches** | **+3000%**   |
| **Total courses**              | 10                 | **211**            | **+2110%**   |
| **Épisodes**                   | 5000               | **10,000**         | **+100%**    |
| **Écart moyen (entraînement)** | 3.39               | ≤2.5               | -26%         |
| **Performance en production**  | gap=2              | **gap=1**          | -50% 🎯      |

---

## ⏱️ TIMELINE

### Phase 1 : Conversion Excel (EN COURS)

```
00:00 - Début de la conversion
07:00 - Géocodage terminé (211 courses)
08:00 - Export JSON complété
```

**Durée estimée** : 7-10 minutes

### Phase 2 : Réentraînement RL (SUIVANT)

```
00:00 - Chargement des ~30 dispatches
00:30 - Début entraînement (10,000 épisodes)
02:30 - Sauvegarde du modèle v2
```

**Durée estimée** : 3-4 heures

### Phase 3 : Déploiement (AUTOMATIQUE)

```
- Modèle sauvegardé : dispatch_optimized_v2.pth
- Remplacement automatique de v1
- Pas de modification de code nécessaire
```

**Durée** : Instantanée

---

## 📊 COMMANDES DE SUIVI

### Monitoring de la Conversion

```bash
# Vérifier la progression
docker exec atmr-api-1 python backend/scripts/monitor_conversion.py

# Logs en temps réel
docker exec atmr-api-1 tail -f data/rl/conversion_output.log

# Dernières lignes
docker exec atmr-api-1 tail -30 data/rl/conversion_output.log
```

### Vérifier le Fichier Généré

```bash
# Voir la taille
docker exec atmr-api-1 ls -lh data/rl/historical_dispatches_from_excel.json

# Compter les dispatches
docker exec atmr-api-1 python -c "
import json
with open('data/rl/historical_dispatches_from_excel.json') as f:
    data = json.load(f)
    print(f'Dispatches: {data[\"total_dispatches\"]}')
    print(f'Bookings: {data[\"total_bookings\"]}')
"
```

---

## 🚀 PROCHAINE ÉTAPE : RÉENTRAÎNEMENT

Une fois la conversion terminée, lancer :

```bash
# Réentraînement avec nouvelles données (10,000 épisodes)
docker exec -d atmr-api-1 bash -c "
cd /app &&
nohup python backend/scripts/rl_train_offline.py > data/rl/training_v2_output.log 2>&1 &
"

# Modifier rl_train_offline.py pour utiliser le nouveau fichier :
# historical_data_file="data/rl/historical_dispatches_from_excel.json"
# num_episodes=10000
# save_path="data/rl/models/dispatch_optimized_v2.pth"
```

---

## 🎯 RÉSULTAT ATTENDU

### Actuellement (Modèle v1)

```
Données      : 1 dispatch
Performance  : gap 3 → 2 (amélioration 33%)
```

### Après Réentraînement (Modèle v2)

```
Données      : ~30 dispatches
Performance  : gap 3 → 1 (amélioration 66%) 🎯
```

**Objectif : Atteindre systématiquement une répartition 3-3-4 ou 4-3-3 !**

---

## 📝 FICHIERS CRÉÉS

1. **`backend/scripts/convert_excel_to_rl_data.py`** (268 lignes)

   - Conversion Excel → JSON RL
   - Géocodage automatique
   - Mapping chauffeurs

2. **`backend/scripts/monitor_conversion.py`** (72 lignes)

   - Suivi de la conversion
   - Statistiques en temps réel

3. **`backend/scripts/list_drivers.py`** (30 lignes)

   - Liste des chauffeurs et leurs initiales

4. **`backend/scripts/analyze_excel.py`** (62 lignes)
   - Analyse de la structure du fichier Excel

---

## ⚠️ NOTES IMPORTANTES

### Géocodage

- **API utilisée** : Nominatim (OpenStreetMap, gratuit)
- **Limite** : 1 requête/seconde (d'où le temps de traitement)
- **Cache** : Les adresses déjà géocodées sont enregistrées
- **Fallback** : Si géocodage échoue → coordonnées par défaut (Genève centre)

### Qualité des Données

Les données Excel contiennent :

- ✅ Adresses complètes (rue, code postal, ville)
- ✅ Heures précises
- ✅ Chauffeurs assignés
- ❌ Pas de coordonnées GPS (on les ajoute via géocodage)
- ❌ Pas de métriques de retard (on les estime à 0)

---

**Dernière mise à jour** : 21 octobre 2025, 00:15  
**Prochaine vérification** : Dans 5 minutes (conversion devrait être terminée)
