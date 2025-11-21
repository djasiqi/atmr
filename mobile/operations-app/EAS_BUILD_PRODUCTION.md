# Guide de Build de Production EAS

Ce guide vous accompagne dans la préparation et l'exécution d'un build de production avec EAS (Expo Application Services).

## 📋 Prérequis

1. **Compte EAS** : Assurez-vous d'être connecté à votre compte EAS

   ```bash
   npm install -g eas-cli
   eas login
   ```

2. **Variables d'environnement** : Configurez les variables nécessaires dans EAS
   ```bash
   eas secret:create --scope project --name EXPO_PUBLIC_API_URL --value https://api.lirie.ch
   eas secret:create --scope project --name EXPO_PUBLIC_ANDROID_MAPS_API_KEY --value YOUR_ANDROID_MAPS_KEY
   eas secret:create --scope project --name SOCKET_HOST --value api.lirie.ch
   eas secret:create --scope project --name SOCKET_PORT --value 5000
   ```

## 🔧 Configuration

### Fichiers de configuration

- **`eas.json`** : Configuration des profils de build (développement, interne, production)
- **`app.config.js`** : Configuration de l'application Expo
- **`package.json`** : Version de l'application

### Variables d'environnement requises

Les variables suivantes doivent être configurées dans EAS Secrets pour le build de production :

| Variable                           | Description                                  | Exemple                  |
| ---------------------------------- | -------------------------------------------- | ------------------------ |
| `APP_VARIANT`                      | Variante de l'application                    | `prod`                   |
| `EXPO_PUBLIC_API_URL`              | URL de l'API backend                         | `https://api.lirie.ch`   |
| `EXPO_PUBLIC_ANDROID_MAPS_API_KEY` | Clé API Google Maps Android                  | `AIza...`                |
| `SOCKET_HOST`                      | Host du serveur WebSocket                    | `api.lirie.ch`           |
| `SOCKET_PORT`                      | Port du serveur WebSocket                    | `5000`                   |
| `GOOGLE_SERVICES_JSON`             | Chemin vers google-services.json (optionnel) | `./google-services.json` |

## 🚀 Build de Production

### Build Android

```bash
# Depuis le répertoire mobile/operations-app
eas build --platform android --profile production
```

Le build Android génère un **App Bundle (AAB)** prêt pour la publication sur Google Play Store.

### Build iOS

```bash
# Depuis le répertoire mobile/operations-app
eas build --platform ios --profile production
```

Le build iOS génère un fichier prêt pour la soumission à l'App Store.

### Build pour les deux plateformes

```bash
eas build --platform all --profile production
```

## 📦 Gestion des versions

### Version de l'application

La version de l'application est définie dans `package.json` :

```json
{
  "version": "1.0.3"
}
```

### Incrémentation automatique

Le profil de production dans `eas.json` est configuré avec `autoIncrement: true`, ce qui signifie que :

- **Android** : Le `versionCode` sera incrémenté automatiquement
- **iOS** : Le `buildNumber` sera incrémenté automatiquement

### Incrémentation manuelle

Pour incrémenter manuellement la version :

```bash
npm run version:patch  # Incrémente la version patch (1.0.3 -> 1.0.4)
```

## 🔐 Credentials et Signing

### Android

Les credentials Android sont gérés automatiquement par EAS. Pour vérifier :

```bash
eas credentials
```

### iOS

Les credentials iOS nécessitent :

- Un compte Apple Developer valide
- Un certificat de distribution
- Un profil de provisioning

Configuration dans `eas.json` :

```json
{
  "submit": {
    "production": {
      "ios": {
        "appleId": "your-apple-id@example.com",
        "ascAppId": "your-app-store-connect-app-id",
        "appleTeamId": "your-apple-team-id"
      }
    }
  }
}
```

## ✅ Checklist avant le build

- [ ] Variables d'environnement configurées dans EAS Secrets
- [ ] Version de l'application mise à jour dans `package.json`
- [ ] Fichier `google-services.json` présent (pour Android)
- [ ] Fichier `GoogleService-Info.plist` présent (pour iOS, si nécessaire)
- [ ] Credentials configurés pour Android et iOS
- [ ] Tests effectués en mode développement
- [ ] Configuration `app.config.js` vérifiée
- [ ] Assets (icônes, splash screen) à jour

## 📱 Soumission aux stores

### Google Play Store

```bash
eas submit --platform android --profile production
```

### Apple App Store

```bash
eas submit --platform ios --profile production
```

## 🔍 Vérification du build

Après le build, vous pouvez :

1. Télécharger le build depuis le dashboard EAS
2. Tester le build sur un appareil physique
3. Vérifier les logs de build pour détecter d'éventuels problèmes

## 🐛 Dépannage

### Erreurs courantes

1. **Variables d'environnement manquantes**
   - Vérifiez que toutes les variables sont définies dans EAS Secrets
   - Utilisez `eas secret:list` pour lister les secrets

2. **Credentials manquants**
   - Exécutez `eas credentials` pour configurer les credentials
   - Suivez les instructions à l'écran

3. **Erreurs de build**
   - Consultez les logs détaillés dans le dashboard EAS
   - Vérifiez la configuration dans `app.config.js`

## 📚 Ressources

- [Documentation EAS Build](https://docs.expo.dev/build/introduction/)
- [Documentation EAS Submit](https://docs.expo.dev/submit/introduction/)
- [Gestion des secrets EAS](https://docs.expo.dev/build-reference/variables/)

## 🔄 Workflow recommandé

1. **Développement** : Utilisez le profil `development` pour tester
2. **Test interne** : Utilisez le profil `internal` pour les tests avec votre équipe
3. **Production** : Utilisez le profil `production` pour les builds finaux

```bash
# Développement
eas build --platform android --profile development

# Test interne
eas build --platform android --profile internal

# Production
eas build --platform android --profile production
```
