# 🔐 Plan de Correction des Secrets Exposés

## ⚠️ URGENT : Secrets exposés dans le dépôt Git

**7 secrets détectés** par GitHub Secret Scanning :

- 5 Google API Keys
- 2 OpenWeather API Keys

## 📋 Fichiers concernés

1. `mobile/operations-app/android/app/src/main/AndroidManifest.xml` (ligne 22)
2. `mobile/operations-app/android/app/src/debug/AndroidManifest.xml` (ligne 22)
3. `mobile/operations-app/android/app/google-services.json` (ligne 18)
4. `mobile/operations-app/google-services.json` (ligne 18)
5. `backend/.env` (lignes 23, 90)
6. `frontend/.env` (probablement)
7. `session/Semaine_4/OPENWEATHER_SETUP.md` (ligne 39) - fichier supprimé ou non trouvé

## 🛠️ Actions à effectuer

### 1. Remplacer les clés dans AndroidManifest.xml

Les clés Google Maps doivent être remplacées par des variables d'environnement ou des placeholders.

**Avant :**

```xml
<meta-data android:name="com.google.android.geo.API_KEY" android:value="AIzaSyA_jC0VzROGO_lEpQg1bicorXYFkOksA-g"/>
```

**Après :**

```xml
<meta-data android:name="com.google.android.geo.API_KEY" android:value="${GOOGLE_MAPS_API_KEY}"/>
```

### 2. Ajouter google-services.json au .gitignore

Ces fichiers contiennent des secrets et ne doivent pas être versionnés.

### 3. Supprimer .env du dépôt Git

Les fichiers `.env` sont déjà dans `.gitignore` mais ont été commités avant. Il faut les supprimer de l'historique Git.

### 4. Révoquer les clés exposées

**⚠️ CRITIQUE :** Toutes les clés exposées doivent être révoquées immédiatement :

- Google Cloud Console → APIs & Services → Credentials
- OpenWeather API Dashboard

## 📝 Commandes à exécuter

```bash
# 1. Supprimer les fichiers sensibles de Git (mais les garder localement)
git rm --cached backend/.env
git rm --cached frontend/.env
git rm --cached mobile/operations-app/android/app/google-services.json
git rm --cached mobile/operations-app/google-services.json

# 2. Ajouter au .gitignore (déjà fait, mais vérifier)
echo "**/google-services.json" >> .gitignore
echo "**/AndroidManifest.xml" >> .gitignore  # Si on veut utiliser des templates

# 3. Créer des fichiers template
cp backend/.env backend/.env.example
cp frontend/.env frontend/.env.example
# Remplacer les secrets par des placeholders dans .env.example

# 4. Commiter les changements
git add .gitignore backend/.env.example frontend/.env.example
git commit -m "Security: Remove exposed secrets from repository"
git push
```

## 🔄 Utilisation de variables d'environnement

### Pour Android (Gradle)

Créer `mobile/operations-app/android/local.properties` :

```properties
GOOGLE_MAPS_API_KEY=YOUR_KEY_HERE
```

Puis dans `build.gradle` :

```gradle
def googleMapsApiKey = project.findProperty("GOOGLE_MAPS_API_KEY") ?: ""
```

### Pour React Native / Expo

Utiliser `app.config.js` avec variables d'environnement :

```javascript
export default {
  extra: {
    googleMapsApiKey: process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY,
  },
  android: {
    config: {
      googleMaps: {
        apiKey: process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY,
      },
    },
  },
};
```

## ✅ Checklist de sécurité

- [ ] Révoquer toutes les clés API exposées
- [ ] Remplacer les clés dans AndroidManifest.xml par des variables
- [ ] Supprimer .env et google-services.json de Git
- [ ] Créer des fichiers .env.example avec placeholders
- [ ] Ajouter google-services.json au .gitignore
- [ ] Vérifier qu'aucun secret n'est dans l'historique Git
- [ ] Configurer les secrets dans GitHub Secrets pour CI/CD
- [ ] Documenter l'utilisation des variables d'environnement
