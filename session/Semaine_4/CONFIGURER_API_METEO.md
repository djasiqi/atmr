# ⚡ CONFIGURER L'API MÉTÉO - GUIDE RAPIDE

**Statut actuel** : ❌ API key non détectée par le container  
**Action requise** : Configurer et redémarrer

---

## 🚀 SOLUTION RAPIDE (2 méthodes)

### Méthode 1 : Script Automatique (Recommandé)

```powershell
# 1. Lancer le script de configuration
docker exec -it atmr-api-1 python scripts/setup_weather_api.py

# 2. Entrer votre clé API quand demandé

# 3. Redémarrer le container
docker-compose restart api

# 4. Vérifier
docker exec atmr-api-1 python -c "import os; print('API Key:', 'OK' if os.getenv('OPENWEATHER_API_KEY') else 'MANQUANTE')"
```

---

### Méthode 2 : Manuel

#### Étape 1 : Créer `backend/.env`

Créer le fichier `backend/.env` avec ce contenu :

```bash
# Configuration OpenWeatherMap API
OPENWEATHER_API_KEY=votre_vraie_cle_ici

# Configuration ML
ML_ENABLED=true
ML_TRAFFIC_PERCENTAGE=10
FALLBACK_ON_ERROR=true
```

⚠️ **Remplacer `votre_vraie_cle_ici` par votre vraie clé !**

#### Étape 2 : Redémarrer

```powershell
docker-compose restart api
```

#### Étape 3 : Vérifier

```powershell
# Test 1 : Variable chargée
docker exec atmr-api-1 python -c "import os; key = os.getenv('OPENWEATHER_API_KEY', ''); print('API Key:', 'CONFIGUREE (' + str(len(key)) + ' chars)' if key else 'MANQUANTE')"

# Test 2 : Service météo
docker exec atmr-api-1 python -c "from services.weather_service import WeatherService; w = WeatherService.get_weather(46.2044, 6.1432); print('Weather factor:', w['weather_factor']); print('Est default:', w.get('is_default', False))"

# Test 3 : Tests complets
docker exec atmr-api-1 python tests/test_weather_service.py
```

---

## ✅ RÉSULTAT ATTENDU

### Avant Configuration

```bash
API Key: MANQUANTE
Weather factor: 0.5
Est default: True  ❌
```

### Après Configuration

```bash
API Key: CONFIGUREE (32 chars)
Weather factor: 0.35  # Valeur réelle variable
Est default: False  ✅
```

---

## 🎯 OBTENIR UNE CLÉ API (GRATUIT)

Si vous n'avez pas encore de clé :

1. **Aller sur** : https://openweathermap.org/
2. **Cliquer** : "Sign Up" (en haut à droite)
3. **Remplir** : Email, Username, Password
4. **Confirmer** : Email (vérifier inbox)
5. **Se connecter** : https://home.openweathermap.org/
6. **Aller dans** : "API keys" (menu)
7. **Copier** : La clé générée automatiquement
8. ⚠️ **Attendre** : 10-15 minutes (activation)

**Format clé** : `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6` (32 caractères)

---

## 🔧 TROUBLESHOOTING

### Problème : "API key not configured"

**Cause** : Container pas redémarré ou `.env` mal placé

**Solution** :

```powershell
# 1. Vérifier fichier existe
Test-Path backend\.env

# 2. Vérifier contenu
Get-Content backend\.env | Select-String "OPENWEATHER"

# 3. Redémarrer (IMPORTANT!)
docker-compose restart api

# 4. Attendre 30s puis tester
Start-Sleep -Seconds 30
docker exec atmr-api-1 python -c "import os; print(os.getenv('OPENWEATHER_API_KEY', 'MANQUANTE')[:10])"
```

### Problème : "401 Unauthorized"

**Cause** : Clé pas encore activée (10-15 min après inscription)

**Solution** : Attendre 15 minutes puis réessayer

### Problème : Clé visible mais `is_default: True`

**Cause** : Clé invalide ou révoquée

**Solution** : Régénérer une nouvelle clé sur openweathermap.org

---

## 📊 IMPACT ATTENDU

Une fois la clé configurée et fonctionnelle :

| Métrique           | Avant (neutre) | Après (réel)        | Gain     |
| ------------------ | -------------- | ------------------- | -------- |
| **R²**             | 0.68           | **0.75+**           | **+11%** |
| **MAE**            | 2.26 min       | **1.80 min**        | **-20%** |
| **Weather factor** | 0.5 (fixe)     | 0.0-1.0 (dynamique) | Variable |

---

## 🎯 VALIDATION FINALE

**Checklist** :

- [ ] backend/.env créé
- [ ] OPENWEATHER_API_KEY ajoutée
- [ ] Container redémarré (`docker-compose restart api`)
- [ ] Variable visible dans container
- [ ] Test `is_default: False` ✅
- [ ] Weather factor variable (pas toujours 0.5)

**Si tous ✅** → API météo fonctionnelle ! 🌦️

---

**📞 Besoin d'aide ?** Consultez `session/Semaine_4/OPENWEATHER_SETUP.md`
