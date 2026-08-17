# P0-E — CAS A affiné + instrumentation LocationTaskConsumer P1–P8

## Formulation exacte

```text
Android / fused system        = frais ✅
Expo task pipeline            = pas de nouvelle location exploitable ❌
→ ne prouve PAS encore que PendingIntent reçoit les fixes
→ sur build ATMR 135, le chemin FGS est déjà LocationCallback (patch)
```

`Finished` ≠ position livrée au JS. SDK / patch :

```text
executeTaskWithLocationBundles(empty)
→ callback.onFinished(null)   // Finished sans locations
→ aucun enqueue
```

## Origine exacte du log observé

Le message :

```text
Location unavailable for foreground-service task delivery
```

n’est **pas** le stock SDK PendingIntent. Il vient du **patch ATMR** :

`onLocationAvailability` quand `isLocationAvailable == false`
(`LocationCallback` FGS path).

Donc le RCA natif doit prioriser :

```text
P2c  onLocationAvailability
P2b  onLocationResult (size / time)
P5   filtre sLastTimestamp
P8   JS=true vs FinishedWithoutLocation
```

PendingIntent P1/P2 restent instrumentés (chemin non-FGS).

## Instrumentation (implémentée dans le source)

Fichier :

- `mobile/unified-app/native-patches/expo-location/LocationTaskConsumer.kt`
- copié vers `node_modules/.../LocationTaskConsumer.kt`

Tag logcat unique : **`ATMR_LTC_P`**

| Id | Checkpoint |
|----|------------|
| P0 | requestLocationUpdates (Callback vs PendingIntent) |
| P1 | didReceiveBroadcast |
| P2 | LocationResult.extractResult (PI) |
| P2b | onLocationResult (FGS Callback) |
| P2c | onLocationAvailability |
| P3 | fallback lastLocation |
| P4 | defer/report (size, hostPaused, shouldReport) |
| P5 | filtre `incoming.time > sLastTimestamp` |
| P6 | scheduleJob / directFGS bundles.size |
| P7 | didExecuteJob bundles.size |
| P8 | executeTask JS=true \| FinishedWithoutLocation |

## Sous-verdicts attendus (run FG instrumenté)

```text
A1  FLP frais + P1=0 + P2b=0          → delivery request / PI-Callback
A2  P2b/P2 null|empty + lastLoc null → FusedLocationProviderClient app
A2' P2c isLocationAvailable=false    → FLP dit unavailable à l’app ★
A3  locations + rejected by P5       → filtre sLastTimestamp ★
A4  accepted + P6/P8 cassés          → TaskManager / execute
```

## Statut figé

```text
P-TECH SYSTEM FLP           = HEALTHY ✅
P-TECH EXPO DELIVERY        = BROKEN ★★★
EXACT NATIVE SUB-LAYER      = OPEN (instrumentation prête)
JS / enqueue / PG / FE      = DOWNSTREAM
IMMUTABILITY 135            = SOUTENU ✅
HOME #3 / Q1 / UX patch     = HOLD ⛔
SERVER                      = inchangé
```

## Next (un seul run)

```text
1. Build interne diag (versionCode+1) avec LTC instrumenté
2. Install témoin 20135
3. FG 60–90 s, filtre :
   adb logcat -s ATMR_LTC_P:I LocationTaskConsumer:W TaskService:I
4. Corréler P2b/P2c/P5/P8 + dumpsys fused_age
5. Trancher A1 / A2' / A3 / A4
```

**Ne pas** lancer ce build EAS tant que GO explicite.

## ✅ Implémenté

- Cadrage CAS A affiné (Finished sans location)
- Instrumentation P0–P8 dans `LocationTaskConsumer.kt` (native-patches + node_modules)
- Tag logcat `ATMR_LTC_P`
- Patch `patches/expo-location+19.0.8.patch` régénéré (exclure `android/build`)
- Doc + plan run FG unique — **EAS build non lancé** (attendre GO)
