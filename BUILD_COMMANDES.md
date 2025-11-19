# 📱 Commandes pour Build APK Debug

## 🚀 Commandes rapides (PowerShell)

### 1. Naviguer vers le dossier Android
```powershell
cd mobile\operations-app\android
```

### 2. Builder l'APK debug
```powershell
.\gradlew.bat assembleDebug
```

### 3. Vérifier l'APK généré
```powershell
Get-ChildItem -Path "app\build\outputs\apk\debug" -Filter "*.apk" | Select-Object Name, @{Name="Taille (MB)";Expression={[math]::Round($_.Length / 1MB, 2)}}, LastWriteTime
```

### 4. Copier l'APK sur le bureau
```powershell
Copy-Item "app\build\outputs\apk\debug\app-debug.apk" -Destination "$env:USERPROFILE\Desktop\app-debug.apk"
```

---

## 📋 Toutes les commandes en une seule fois

```powershell
cd mobile\operations-app\android; .\gradlew.bat assembleDebug; Get-ChildItem -Path "app\build\outputs\apk\debug" -Filter "*.apk" | Select-Object -First 1 | ForEach-Object { Write-Host "✅ APK : $($_.FullName)"; Write-Host "📦 Taille : $([math]::Round($_.Length / 1MB, 2)) MB"; Write-Host "📅 Date : $($_.LastWriteTime)" }
```

---

## 🔧 Commandes utiles supplémentaires

### Installer directement sur appareil connecté
```powershell
cd mobile\operations-app\android
.\gradlew.bat installDebug
```

### Nettoyer le build (si problème)
```powershell
cd mobile\operations-app\android
.\gradlew.bat clean
.\gradlew.bat assembleDebug
```

### Voir l'emplacement exact de l'APK
```powershell
cd mobile\operations-app\android
Resolve-Path "app\build\outputs\apk\debug\app-debug.apk"
```

---

## 📍 Emplacement de l'APK

L'APK sera généré dans :
```
mobile\operations-app\android\app\build\outputs\apk\debug\app-debug.apk
```

Chemin complet :
```
C:\Users\jasiq\atmr\mobile\operations-app\android\app\build\outputs\apk\debug\app-debug.apk
```

