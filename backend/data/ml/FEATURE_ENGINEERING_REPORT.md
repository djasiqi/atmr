# 🔧 RAPPORT DE FEATURE ENGINEERING

## 📊 RÉSUMÉ

- **Features originales** : 17
- **Features après engineering** : 40
- **Nouvelles features créées** : 23

## 🆕 NOUVELLES FEATURES CRÉÉES

### Interactions

- `distance_x_traffic`
- `distance_x_weather`
- `traffic_x_weather`
- `medical_x_distance`
- `urgent_x_traffic`

### Temporelles

- `is_rush_hour`
- `is_morning_peak`
- `is_evening_peak`
- `hour_sin`
- `hour_cos`
- `is_weekend`
- `day_sin`
- `day_cos`
- `is_lunch_time`

### Agrégées

- `delay_by_hour`
- `delay_by_day`
- `driver_experience_level`
- `delay_by_driver_exp`
- `distance_category`
- `traffic_level`

### Polynomiales

- `distance_squared`
- `traffic_squared`
- `driver_exp_log`

---

**Rapport généré automatiquement par `feature_engineering.py`**
