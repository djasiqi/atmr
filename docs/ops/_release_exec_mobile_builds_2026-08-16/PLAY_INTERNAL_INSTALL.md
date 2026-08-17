# Install Play Internal Testing — build 126

```text
EAS build id     = a2970b22-6b5c-4390-a81b-74588e06b50b
EAS submission   = 427b7707-140d-4910-97d0-c78297a93dc3
URL              = https://expo.dev/accounts/drinjasiqi/projects/operations-app/submissions/427b7707-140d-4910-97d0-c78297a93dc3
App version      = 1.0.11
versionCode      = 126
Track            = internal (COMPLETED)
General prod     = NO-GO (ne pas promouvoir)
Submit result    = Submitted your app to Google Play Store! (2026-08-16)
```

## Checklist install device (manuel / testeur)

1. Play Console → Testing → Internal testing : confirmer release **126** disponible.
2. Compte Google du Samsung `RFCW20QC53W` dans la liste de testeurs Internal + lien d’opt-in accepté.
3. Sur le device :
   - désinstaller le sideload `ch.liri.operations` (signature debug ≠ Play),
   - ouvrir Play Store → app Lirie / operations → **Mettre à jour / Installer** depuis Internal Testing,
   - **ne pas** sideload l’APK local.
4. Vérifier via adb :

```powershell
adb -s RFCW20QC53W shell dumpsys package ch.liri.operations | findstr version
adb -s RFCW20QC53W shell cmd package get-installer-package-name ch.liri.operations
```

Attendu :

```text
versionName=1.0.11
versionCode=126
installer ≈ com.android.vending
```

5. Lancer l’app depuis le launcher (pas Metro).
6. Login chauffeur smoke + mission `IN_PROGRESS`.
7. Lancer le smoke :

```powershell
.\docs\ops\_release_exec_mobile_builds_2026-08-16\run_smoke_play_internal.ps1 -AdbSerial RFCW20QC53W -DriverId 20135
```

## Statut

```text
PLAY SUBMIT INTERNAL   = DONE ✅
PLAY INSTALL ON DEVICE = DONE ✅
SMOKE PLAY             = FAIL ❌ (= sideload ; P0-D OPEN)
GENERAL DISTRIBUTION   = NO-GO ❌
```

Rapport : `SMOKE_PLAY_INTERNAL_REPORT.md`  
RCA : `docs/ops/gps-p0-d-android-prod-binary-fgs-bg-2026-08-16.md`
