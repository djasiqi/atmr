# 🤖 RAPPORT D'ENTRAÎNEMENT DU MODÈLE ML

## 📊 MÉTRIQUES DE PERFORMANCE

### Test Set

- **MAE** : 2.26 min ✅
- **RMSE** : 2.84 min
- **R²** : 0.6757 ✅
- **Temps prédiction** : 34.07ms ✅

### Validation Croisée (5-Fold)

- **MAE (CV)** : 2.17 ± 0.05 min
- **R² (CV)** : 0.6681 ± 0.0196
- **Stabilité** : 0.0196 ✅

### Overfitting Check

- **Diff R² (train - test)** : 0.2784
- ⚠️ **Overfitting détecté**

## 🎯 TOP 10 FEATURES

| Rang | Feature | Importance |
|------|---------|------------|
| 14 | `distance_x_weather` | 0.3473 |
| 15 | `traffic_x_weather` | 0.1898 |
| 4 | `distance_km` | 0.0700 |
| 33 | `distance_squared` | 0.0615 |
| 10 | `driver_total_bookings` | 0.0504 |
| 35 | `driver_exp_log` | 0.0491 |
| 13 | `distance_x_traffic` | 0.0491 |
| 12 | `weather_factor` | 0.0315 |
| 5 | `duration_seconds` | 0.0259 |
| 3 | `month` | 0.0180 |

---

**Rapport généré automatiquement par `train_model.py`**
