# 🌦️ CONFIGURATION OPENWEATHERMAP API

**Service** : OpenWeatherMap  
**Plan** : Free (1,000 calls/jour)  
**Coût** : €0 / mois

---

## 📝 INSCRIPTION

### Étape 1 : Créer un Compte

1. Aller sur https://openweathermap.org/
2. Cliquer sur "Sign Up" (en haut à droite)
3. Remplir le formulaire :
   - Email
   - Username
   - Password
4. Confirmer email (vérifier inbox)

### Étape 2 : Obtenir l'API Key

1. Se connecter sur https://home.openweathermap.org/
2. Aller dans "API keys" (menu)
3. Copier la clé générée automatiquement
   - Exemple : `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`
4. ⚠️ **Attendre 10-15 minutes** (activation clé)

---

## ⚙️ CONFIGURATION

### Backend (.env)

Ajouter dans `backend/.env` :

```bash
# OpenWeatherMap API
OPENWEATHER_API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

### Docker (docker-compose.yml)

Ou ajouter dans `docker-compose.yml` :

```yaml
services:
  api:
    environment:
      - OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY}
```

Puis créer `.env` à la racine :

```bash
OPENWEATHER_API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

### Redémarrer

```bash
# Redémarrer le container
docker-compose restart api

# Vérifier la variable
docker exec atmr-api-1 printenv | grep OPENWEATHER
```

---

## 🧪 TESTER L'API

### Test 1 : Via Python

```bash
docker exec atmr-api-1 python -c "
from services.weather_service import WeatherService

# Genève (46.2044, 6.1432)
weather = WeatherService.get_weather(46.2044, 6.1432)

print(f'Température: {weather[\"temperature\"]}°C')
print(f'Conditions: {weather[\"main_condition\"]}')
print(f'Weather Factor: {weather[\"weather_factor\"]:.2f}')
print(f'Est défaut: {weather.get(\"is_default\", False)}')
"
```

**Résultat attendu** :

```
Température: 12.5°C
Conditions: Clouds
Weather Factor: 0.35
Est défaut: False
```

### Test 2 : Via Tests

```bash
# Installer requests si nécessaire
docker exec atmr-api-1 pip install requests

# Lancer tests
docker exec atmr-api-1 pytest tests/test_weather_service.py -v
```

---

## 📊 LIMITES & QUOTAS

### Plan Free

| Limite           | Valeur |
| ---------------- | ------ |
| **Calls/jour**   | 1,000  |
| **Calls/minute** | 60     |
| **Coût**         | €0     |

### Notre Usage

**Estimation** :

- 100-150 bookings/jour
- 1 call API par booking = **100-150 calls/jour**
- Avec cache 1h = **~50 calls/jour** ✅

**Conclusion** : ✅ **Largement sous la limite**

### Si Dépassement

**Symptôme** :

```
ERROR [Weather] API call failed: 429 Too Many Requests
```

**Solution** :

1. Augmenter TTL cache (1h → 2h)
2. Fallback neutre automatique
3. Upgrader plan si nécessaire ($0.90/mois pour 100k calls)

---

## 🔧 TROUBLESHOOTING

### Problème 1 : "API key not configured"

**Symptôme** :

```
WARNING [Weather] API key not configured, using default factor
```

**Solution** :

1. Vérifier `.env` : `OPENWEATHER_API_KEY=...`
2. Redémarrer container
3. Vérifier variable : `docker exec atmr-api-1 printenv | grep OPENWEATHER`

### Problème 2 : "Invalid API key"

**Symptôme** :

```
ERROR [Weather] API call failed: 401 Unauthorized
```

**Solutions** :

1. Attendre 10-15 min (activation clé)
2. Régénérer clé sur openweathermap.org
3. Vérifier copier/coller correct (pas d'espaces)

### Problème 3 : Timeout

**Symptôme** :

```
ERROR [Weather] API call failed: timeout
```

**Solutions** :

1. Vérifier connexion internet
2. Utilise fallback neutre automatique (0.5)
3. Augmenter timeout (5s → 10s)

---

## ✅ VALIDATION

### Checklist

- [ ] Compte OpenWeatherMap créé
- [ ] API key générée
- [ ] `.env` configuré
- [ ] Container redémarré
- [ ] Variable visible (`printenv`)
- [ ] Test Python OK
- [ ] Weather factor != 0.5 (pas default)

**Si tous ✅** → API météo opérationnelle ! 🌦️

---

## 💡 IMPACT ATTENDU

### Avant (weather_factor = 0.5)

- Weather neutre partout
- Pas de différenciation conditions
- R² 0.6757

### Après (weather_factor réel)

- Conditions réelles temps réel
- Facteurs météo importants : **53.7%**
- **R² attendu : 0.75+** (+11%)
- **MAE attendu : 1.80 min** (-20%)

---

**📞 Support OpenWeatherMap** : https://openweathermap.org/faq
