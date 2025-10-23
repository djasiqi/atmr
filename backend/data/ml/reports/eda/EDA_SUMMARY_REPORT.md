# 📊 RAPPORT D'ANALYSE EXPLORATOIRE (EDA)

**Dataset** : 5000 échantillons × 17 features

---

## 📈 STATISTIQUES DESCRIPTIVES

### Target: `actual_delay_minutes`

- **Moyenne** : 6.28 min
- **Médiane** : 5.78 min
- **Écart-type** : 4.83 min
- **Min / Max** : -6.52 / 57.48 min
- **Q1 / Q3** : 3.15 / 8.70 min

## 🔗 CORRÉLATIONS PRINCIPALES

| Feature | Corrélation | Force |
|---------|-------------|-------|
| `distance_km` | +0.619 | Forte |
| `duration_seconds` | +0.585 | Forte |
| `traffic_density` | +0.357 | Moyenne |
| `weather_factor` | +0.294 | Faible |
| `driver_total_bookings` | -0.199 | Faible |
| `day_of_week` | -0.140 | Faible |

## 🔍 OUTLIERS DÉTECTÉS

**Méthode IQR** : 138 outliers (2.76%)

- Borne inférieure : -5.17
- Borne supérieure : 17.02

**Méthode Z-score** : 63 outliers (1.26%)

## 💡 INSIGHTS & RECOMMANDATIONS

### Points Clés

1. **Feature la plus prédictive** : `distance_km` (corr: +0.619)

### Prochaines Étapes

1. **Feature Engineering** : Créer interactions entre top features
2. **Traitement Outliers** : Décider de conserver ou transformer
3. **Normalisation** : Préparer features pour ML
4. **Split Train/Test** : 80/20 avec stratification

---

**Rapport généré automatiquement par `analyze_data.py`**
