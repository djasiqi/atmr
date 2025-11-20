# Gestion des Clés API et Secrets

Ce document explique comment gérer et changer les clés API exposées dans le dépôt.

## ⚠️ Clés à Changer

### 1. Google Maps API Key (Mobile)

**Fichiers concernés :**
- `mobile/operations-app/android/app/src/main/AndroidManifest.xml`
- `mobile/operations-app/google-services.json` (déjà dans .gitignore)

**Solution :**
La clé Google Maps est maintenant injectée automatiquement depuis la variable d'environnement `EXPO_PUBLIC_ANDROID_MAPS_API_KEY` lors du prebuild Expo.

**Étapes pour changer la clé :**

1. Créer ou modifier le fichier `.env` dans `mobile/operations-app/` :
```bash
EXPO_PUBLIC_ANDROID_MAPS_API_KEY=votre_nouvelle_cle_google_maps
```

2. Régénérer le projet Android avec :
```bash
cd mobile/operations-app
npx expo prebuild --clean
```

3. Vérifier que le fichier `android/app/src/main/AndroidManifest.xml` contient la nouvelle clé.

**Note :** Le fichier `google-services.json` contient également une clé API Firebase. Ce fichier est déjà ignoré par Git (voir `.gitignore` ligne 80). Assurez-vous de ne pas le committer.

### 2. OpenWeather API Key (Backend)

**Fichier concerné :**
- `backend/.env` (ligne 90, variable `OPENWEATHER_API_KEY`)

**Solution :**
La clé OpenWeather est déjà lue depuis la variable d'environnement `OPENWEATHER_API_KEY` dans `backend/services/weather_service.py`.

**⚠️ ACTION URGENTE REQUISE :**

La clé actuelle `68700f6462b4c098e4d1a10c041378c6` a été exposée publiquement et doit être **immédiatement révoquée et remplacée**.

**Étapes pour changer la clé :**

1. **Révoquer l'ancienne clé** sur [OpenWeatherMap](https://openweathermap.org/api) :
   - Connectez-vous à votre compte OpenWeatherMap
   - Allez dans "API keys"
   - Révoquez la clé `68700f6462b4c098e4d1a10c041378c6`

2. **Générer une nouvelle clé** :
   - Créez une nouvelle clé API sur OpenWeatherMap
   - Notez la nouvelle clé

3. **Mettre à jour le fichier `backend/.env`** :
```bash
# Ouvrir backend/.env et modifier la ligne 90
OPENWEATHER_API_KEY=votre_nouvelle_cle_openweather
```

4. **Redémarrer le serveur backend** pour appliquer les changements :
```bash
# Selon votre méthode de démarrage
# Si Docker:
docker-compose restart backend

# Si directement:
# Arrêter le serveur (Ctrl+C) et redémarrer
python backend/app.py
```

5. **Vérifier que la nouvelle clé fonctionne** :
   - Testez une requête météo via l'API
   - Vérifiez les logs pour confirmer qu'il n'y a pas d'erreur d'authentification

**Note :** 
- Le fichier `.env` est déjà ignoré par Git (voir `.gitignore` ligne 60-63)
- Un fichier `backend/env.example` est maintenant disponible comme template
- Pour créer votre `.env`, copiez le template : `cp backend/env.example backend/.env`
- Assurez-vous de ne jamais committer le fichier `.env` avec les vraies clés

## 🔒 Bonnes Pratiques

### Fichiers à NE JAMAIS committer :
- `backend/.env`
- `mobile/operations-app/.env`
- `mobile/operations-app/google-services.json`
- `mobile/operations-app/android/app/google-services.json`
- Tous les fichiers contenant des secrets

### Fichiers de référence :
- `mobile/operations-app/env.example` - Template pour les variables d'environnement mobile
- `backend/.env.example` - Template pour les variables d'environnement backend

### Vérification avant commit :
```bash
# Vérifier qu'aucun secret n'est dans les fichiers trackés
git diff --cached | grep -i "api_key\|secret\|password\|token"
```

## 🔄 Régénération après changement de clés

### Mobile (Expo)
```bash
cd mobile/operations-app
# Supprimer le dossier android généré
rm -rf android
# Régénérer avec les nouvelles variables d'environnement
npx expo prebuild --clean
```

### Backend
```bash
cd backend
# Redémarrer le serveur pour charger les nouvelles variables
# (selon votre méthode de démarrage)
```

## 📝 Variables d'environnement requises

### Mobile (`mobile/operations-app/.env`)
- `EXPO_PUBLIC_ANDROID_MAPS_API_KEY` - Clé Google Maps pour Android
- `EXPO_PUBLIC_GOOGLE_API_KEY` - Clé Google API REST (Directions, etc.)

### Backend (`backend/.env`)
- `OPENWEATHER_API_KEY` - Clé OpenWeatherMap API

## 🚨 En cas de clé exposée

Si une clé a été exposée publiquement :

1. **Révoquer immédiatement la clé** dans la console du fournisseur (Google Cloud, OpenWeather, etc.)
2. **Générer une nouvelle clé**
3. **Mettre à jour tous les environnements** (dev, staging, production)
4. **Vérifier les logs** pour détecter tout usage non autorisé
5. **Surveiller les facturations** pour détecter des abus

