# 📚 Guide de référence - Commandes essentielles

Ce document contient toutes les commandes essentielles pour gérer le serveur, Git, et les builds Android.

---

## 🖥️ SERVEUR - Accès et déploiement

### Accès au serveur

```bash
ssh deploy@138.201.155.201
```

### Vérifier l'état Docker

```bash
# Voir les conteneurs en cours d'exécution
docker ps

# Voir tous les conteneurs (y compris arrêtés)
docker ps -a

# Voir l'état des services docker-compose
cd /srv/atmr/backend/backend
docker compose ps

# Voir les logs d'un conteneur
docker logs backend-api-1
docker logs backend-api-1 --tail 50 -f  # Suivre les logs en temps réel
```

### Redémarrer les services

```bash
cd /srv/atmr/backend/backend

# Redémarrer un service spécifique
docker compose restart api

# Redémarrer tous les services
docker compose restart

# Arrêter et redémarrer
docker compose down
docker compose up -d
```

### Redéployer la dernière image

```bash
cd /srv/atmr/backend/backend

# Récupérer la dernière image depuis Docker Hub
docker compose pull api

# Redémarrer avec la nouvelle image
docker compose up -d api

# Ou rebuild et redémarrer
docker compose up -d --build api
```

### Commandes Docker utiles

```bash
# Voir l'utilisation des ressources
docker stats

# Nettoyer les images/containers inutilisés
docker system prune -a

# Voir les logs de tous les services
docker compose logs -f

# Voir les logs d'un service spécifique avec filtrage
docker logs backend-api-1 2>&1 | grep -i "erreur\|error\|exception"
```

---

## 📦 GIT - Commandes essentielles

### Configuration de base

```bash
# Vérifier le statut
git status

# Voir les différences
git diff

# Voir l'historique
git log --oneline -10
```

### Ajouter et committer

```bash
# Ajouter tous les fichiers modifiés
git add -A

# Ajouter un fichier spécifique
git add chemin/vers/fichier

# Committer avec message
git commit -m "Description des changements"

# Committer tous les fichiers modifiés (sans add)
git commit -a -m "Description"
```

### Push et Pull

```bash
# Pousser vers le dépôt distant
git push

# Pousser vers une branche spécifique
git push origin main

# Récupérer les dernières modifications
git pull

# Récupérer sans merger
git fetch
```

### Branches

```bash
# Voir les branches
git branch

# Créer une nouvelle branche
git checkout -b nom-branche

# Changer de branche
git checkout nom-branche

# Merger une branche
git merge nom-branche
```

### Annuler des changements

```bash
# Annuler les modifications non commitées
git restore fichier

# Annuler tous les changements non commités
git restore .

# Annuler le dernier commit (garder les fichiers)
git reset --soft HEAD~1

# Annuler le dernier commit (supprimer les fichiers)
git reset --hard HEAD~1
```

---

## 📱 ANDROID - Build développement (debug, local)

### Prérequis

```bash
# Depuis la racine du projet
cd mobile/operations-app
```

### Configuration pour développement

**PowerShell:**

```powershell
$env:APP_VARIANT = "dev"
$env:NODE_ENV = "development"
$env:EXPO_PUBLIC_API_URL = "http://localhost:5000"
```

**CMD:**

```cmd
set APP_VARIANT=dev
set NODE_ENV=development
set EXPO_PUBLIC_API_URL=http://localhost:5000
```

### Régénérer les fichiers natifs

```bash
# Nettoyer et régénérer les fichiers Android
npx expo prebuild --platform android --clean
```

### Build APK debug (pour téléchargement local)

```bash
cd android

# Build APK debug (non signé, pour test local)
.\gradlew.bat assembleDebug

# L'APK sera dans :
# android/app/build/outputs/apk/debug/app-debug.apk
```

### Installer sur appareil connecté

```bash
# Installer directement sur appareil connecté via USB
.\gradlew.bat installDebug

# Ou via ADB
adb install app\build\outputs\apk\debug\app-debug.apk
```

### Vérifier l'APK généré

```bash
# Voir où se trouve l'APK
dir app\build\outputs\apk\debug\app-debug.apk

# Copier vers un emplacement accessible
copy app\build\outputs\apk\debug\app-debug.apk %USERPROFILE%\Desktop\app-debug.apk
```

---

## 🚀 ANDROID - Build production (Play Store)

### Configuration pour production

**PowerShell:**

```powershell
cd mobile/operations-app/android
$env:NODE_ENV = "production"
$env:EXPO_PUBLIC_API_URL = "https://api.lirie.ch"
```

**CMD:**

```cmd
cd mobile/operations-app/android
set NODE_ENV=production
set EXPO_PUBLIC_API_URL=https://api.lirie.ch
```

### Vérifier la configuration keystore

```bash
# Vérifier que le keystore existe
dir app\upload-keystore.jks

# Vérifier les informations du keystore (optionnel)
keytool -list -v -keystore app\upload-keystore.jks -storepass "mot_de_passe"
```

### Mettre à jour le versionCode (si nécessaire)

```bash
# Éditer le fichier
# mobile/operations-app/android/app/build.gradle
# Ligne 95: versionCode X  (doit être > dernière version Play Store)
# Ligne 96: versionName "X.X.X"
```

### Build AAB pour Play Store

```bash
# Build Android App Bundle (recommandé pour Play Store)
.\gradlew.bat bundleRelease

# L'AAB sera dans :
# android/app/build/outputs/bundle/release/app-release.aab
```

### Build APK release (alternative)

```bash
# Build APK release (si besoin d'un APK signé)
.\gradlew.bat assembleRelease

# L'APK sera dans :
# android/app/build/outputs/apk/release/app-release.apk
```

### Vérifier et copier l'AAB/APK

```bash
# Vérifier que l'AAB existe
dir app\build\outputs\bundle\release\app-release.aab

# Copier vers le bureau pour faciliter l'upload
copy app\build\outputs\bundle\release\app-release.aab %USERPROFILE%\Desktop\app-release.aab
```

### Nettoyer le build (si problème)

```bash
cd android
.\gradlew.bat clean
.\gradlew.bat bundleRelease
```

---

## ✅ CHECKLIST RAPIDE - Build production

```bash
# 1. Vérifier les variables d'environnement
set NODE_ENV=production
set EXPO_PUBLIC_API_URL=https://api.lirie.ch

# 2. Vérifier le keystore
dir app\upload-keystore.jks

# 3. Vérifier le versionCode dans build.gradle
# (doit être supérieur à la dernière version sur Play Store)

# 4. Builder
cd android
.\gradlew.bat bundleRelease

# 5. Vérifier l'AAB généré
dir app\build\outputs\bundle\release\app-release.aab

# 6. Copier pour upload
copy app\build\outputs\bundle\release\app-release.aab %USERPROFILE%\Desktop\
```

---

## 🔧 DÉPANNAGE

### Android - Erreurs de build

```bash
# Nettoyer complètement
cd android
.\gradlew.bat clean

# Vérifier les logs détaillés
.\gradlew.bat bundleRelease --stacktrace

# Vérifier les logs avec plus d'infos
.\gradlew.bat bundleRelease --info
```

### Docker - Problèmes sur le serveur

```bash
# Voir les logs en temps réel avec filtrage
docker logs -f backend-api-1 2>&1 | grep -i "erreur\|error\|exception"

# Redémarrer un service spécifique
docker compose restart api

# Voir l'état de santé
docker compose ps

# Voir l'utilisation des ressources
docker stats
```

### Git - Problèmes courants

```bash
# Voir les changements non commités
git status
git diff

# Annuler tous les changements locaux
git restore .

# Forcer le push (attention!)
git push --force
```

---

## 📝 NOTES IMPORTANTES

### Keystore

- **Emplacement**: `mobile/operations-app/android/app/upload-keystore.jks`
- **Ne JAMAIS commiter le keystore** (déjà dans `.gitignore`)
- **Sauvegarder le keystore** dans un endroit sûr
- **Empreinte SHA1 attendue**: `9F:50:84:8B:7A:BD:1E:83:35:A4:55:F0:70:99:27:27:3A:17:77:92`

### VersionCode

- Doit être **strictement supérieur** à la dernière version sur Play Store
- Vérifier dans Play Console → Production → Releases
- Modifier dans `mobile/operations-app/android/app/build.gradle` ligne 95

### Variables d'environnement

- **Développement**: `APP_VARIANT=dev`, `NODE_ENV=development`
- **Production**: `NODE_ENV=production`, `EXPO_PUBLIC_API_URL=https://api.lirie.ch`
- Les variables sont valides uniquement pour la session en cours

### EAS Build

- Pour télécharger le keystore depuis EAS:
  ```bash
  eas login
  eas credentials
  # Sélectionner Android → production → Download existing keystore
  ```

---

## 🔗 LIENS UTILES

- **Play Store Console**: https://play.google.com/console
- **EAS Dashboard**: https://expo.dev
- **Serveur**: `138.201.155.201` (ssh deploy@138.201.155.201)

---

_Dernière mise à jour: 2025-11-17_
