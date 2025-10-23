# 📊 SEMAINE 3 - ML COLLECTE & PRÉPARATION DONNÉES

**Focus** : Machine Learning - Foundation & Data Preparation  
**Durée** : 5 jours (Lundi à Vendredi)  
**Difficulté** : ⭐⭐⭐ Moyenne-Avancée

---

## 🎯 OBJECTIF PRINCIPAL

Préparer l'infrastructure et les données pour implémenter le Machine Learning dans le système de dispatch, en créant un modèle de prédiction des retards basé sur des données historiques réelles.

---

## 📋 VUE D'ENSEMBLE

Cette semaine se concentre sur 3 axes principaux :

### 1️⃣ **Collecte de Données** (Lundi)

- Extraction des données historiques (90 jours)
- Feature engineering de base
- Création du dataset brut

### 2️⃣ **Analyse & Préparation** (Mardi-Mercredi)

- Analyse exploratoire (EDA)
- Feature engineering avancé
- Nettoyage et normalisation

### 3️⃣ **Modélisation** (Jeudi-Vendredi)

- Entraînement modèle baseline
- Intégration dans le dispatch
- Tests et validation

---

## 🎯 OBJECTIFS CHIFFRÉS

| Métrique                      | Cible                | Critique     |
| ----------------------------- | -------------------- | ------------ |
| **Dataset size**              | > 5,000 échantillons | ✅ Oui       |
| **Features**                  | 30+ features         | ✅ Oui       |
| **MAE** (Mean Absolute Error) | < 5 minutes          | ✅ Oui       |
| **R² score**                  | > 0.6                | ⚠️ Désirable |
| **Temps prédiction**          | < 100ms              | ✅ Oui       |

---

## 🗓️ PLANNING DÉTAILLÉ

### 📅 JOUR 1 (LUNDI) - Collecte de Données

**Objectif** : Extraire et préparer les données historiques

**Tâches** :

- Créer `backend/scripts/ml/collect_training_data.py`
- Extraire bookings + assignments des 90 derniers jours
- Feature engineering de base (15-20 features)
- Sauvegarder en CSV et JSON

**Temps estimé** : 6h  
**Livrable** : `training_data.csv` avec 5,000-10,000 lignes

---

### 📅 JOUR 2 (MARDI) - Analyse Exploratoire (EDA)

**Objectif** : Comprendre les données et identifier patterns

**Tâches** :

- Pandas Profiling Report automatique
- Distribution des retards (histogrammes)
- Matrice de corrélation (heatmap)
- Détection outliers et anomalies
- Visualisations temporelles

**Temps estimé** : 6h  
**Livrable** : `data_analysis_report.html` avec insights

---

### 📅 JOUR 3 (MERCREDI) - Feature Engineering Avancé

**Objectif** : Enrichir les features pour un modèle performant

**Tâches** :

- Ajouter 15+ features supplémentaires :
  - Historique performance driver (7 derniers jours)
  - Patterns temporels (heure, jour, mois)
  - Données météo (optionnel si API)
  - Distance vs durée OSRM ratio
  - Indicateurs de pooling
- Normalisation (StandardScaler)
- Encoding catégories (OneHotEncoder)
- Train/test split (80/20)

**Temps estimé** : 6h  
**Livrable** : `extract_features_v2()` fonction

---

### 📅 JOUR 4 (JEUDI) - Entraînement Modèle

**Objectif** : Créer un modèle ML baseline fonctionnel

**Tâches** :

- Entraîner RandomForestRegressor
- Validation croisée (5-fold CV)
- Calcul métriques (MAE, RMSE, R²)
- Feature importance analysis
- Hyperparameter tuning (GridSearchCV)
- Sauvegarder modèle et scaler

**Temps estimé** : 6h  
**Livrable** : `model_rf.pkl` + `scaler.pkl` avec MAE < 5min

---

### 📅 JOUR 5 (VENDREDI) - Intégration & Tests

**Objectif** : Activer le ML dans le système de dispatch

**Tâches** :

- Activer `ml_predictor.py` (code existant)
- Tests de prédiction en temps réel
- Comparaison ML vs baseline
- Logging des prédictions
- Documentation pipeline complet
- Rapport final Semaine 3

**Temps estimé** : 6h  
**Livrable** : ML actif en production

---

## 📊 IMPACT ATTENDU

| Métrique                      | Sans ML  | Avec ML  | Amélioration    |
| ----------------------------- | -------- | -------- | --------------- |
| **Retards prévisibles**       | 0%       | 70-80%   | ✅ Anticipation |
| **Réassignations proactives** | 0        | ~20/jour | ✅ Optimisation |
| **Satisfaction client**       | Baseline | +15-20%  | ✅ Proactivité  |
| **Coûts opérationnels**       | Baseline | -10-15%  | ✅ Efficacité   |

---

## ✅ CHECKLIST DE DÉMARRAGE

- [ ] Semaine 2 terminée et validée
- [ ] PostgreSQL avec données historiques (90 jours)
- [ ] Python packages installés (pandas, scikit-learn)
- [ ] Dossier `backend/scripts/ml/` créé
- [ ] Documentation lue (README + GUIDE)
- [ ] Prêt à coder ! 🚀

---

**Prochaine étape** : Lire le **GUIDE_DETAILLE.md** ! 📖
