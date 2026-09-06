# DRIVER COLD START UX — stabilité visuelle

**Statut figé** — ne pas fermer 01 sur le seul Dev Client. Ne pas rouvrir 02.

```text
DRIVER-COLD-01
  JS BootSurface              PASS
  JS overlay → hub            PASS
  Native splash branding      FAIL → splash-solid remplacé en config ;
                              rebuild natif obligatoire
  Dev Client → Metro white    DEV ARTIFACT
                              → ne pas corriger dans React
                              → ne pas conclure sur la prod
  P0 GPS LOOP                 CLOSED
                              (distinct de l’écran blanc)
  VISUAL COLD METRO           FAIL CONFIRMED (20:45, 3095 modules / 39,9 s)
  PRODUCT DEVICE GATE         PENDING
                              → smoke APK standalone / preview
                              → PAS DevLauncher, PAS Metro

DRIVER-COLD-02 = PASS / CLOSED
DRIVER-COLD-03 = ligne GPS/statut unique (plus de réserve 48 px)
DRIVER-COLD-04 = BLOCKED BY MAP CONFIG

DRIVER-RUNTIME-01 = PASS
P0 GPS            = CLOSED
DRIVER-QUEUE-409-01 = QUALIFIÉ / BLOQUEUR POTENTIEL
DEPLOY            = TOUJOURS BLOQUÉ

DRIVER HUB SPACING
  CURRENT = FAIL VISUEL → CORRECTIF IMPLÉMENTÉ
  DEVICE GATE = PENDING
  TARGET:
    ligne 3 = GPS actif OU alerte
    aucune height:48 StatusArea
    gps/status → missionTop = 24–32 px
    missionTop identique avec/sans alerte
```

```text
NO GPS LOGIC CHANGE
NO TRACKING CADENCE CHANGE
NO BATTERY / FGS CHANGE
NO SELF-HEAL CHANGE
NO CAMERA POLICY CHANGE tant que la carte DEV est beige
```

## Sauts visibles

```text
P0-A  Splash natif → BootSplash → hub     01 JS PASS / natif FAIL / gate produit PENDING
P0-B  Skeleton → mission/idle             DRIVER-COLD-02 PASS / CLOSED
P0-C  Carte → premier fix GNSS            DRIVER-COLD-04 BLOCKED BY MAP
```

## Lecture du smoke (2026-09-06)

Metro ~20 s en DEV n’est **pas** un FAIL. Le défaut 01 est le flash et l’absence de logo :

```text
menthe → blanc → menthe → blanc → overlay pâle sans logo
```

| ID | Verdict smoke | Lecture |
|---|---|---|
| 01 | **FAIL** | Succession blanc ↔ menthe, splash/overlay sans logo |
| 02 | **PASS / CLOSED** | Shell, header, mission stables une fois le hub peint |
| 03 | pas FAIL | Chip à t+38 s dans la zone 48 px — `mission.y` non mesuré |
| 04 | pas un FAIL caméra | Pas de tuiles, pas de marker, pas de premier fix observable |

## DRIVER-COLD-01 — Splash continuity

**Statut** : JS **PASS** · splash natif **FAIL** (config corrigée, rebuild requis) · Metro blanc = **DEV ARTIFACT** · gate produit **PENDING**.

Continuum produit (build standalone, bundle embarqué) :

```text
ANDROID SPLASH
#EAF3F1 + logo LIRIE (220 dp)
        ↓
JS BootSurface
#EAF3F1 + logo LIRIE (220×95)
        ↓
fade
        ↓
Hub chauffeur
```

Ne plus rejouer DRIVER-COLD-01 sur le Dev Client après ce correctif natif : on retomberait sur le bruit Metro (`Loading from 127.0.0.1:8081…`), qui n’existe pas en prod.

```text
[x] hideAsync n’est plus déclenché sur fonts seules
[x] overlay #EAF3F1 peint avant hide natif
[x] Lottie skippé au cold start suivant, overlay conservé jusqu’au shell
[x] pas de « Chargement de la session… » si session locale valide
[x] guards se résolvent derrière l’overlay
[x] plus de `return null` pendant le chargement des fonts (root RN blanc)
[x] GestureHandlerRootView / SafeArea / Stack contentStyle = #EAF3F1
[x] overlay affiche toujours le logo (Lottie 1er launch, wordmark sinon)
[x] hold + surface derrière Redirect = BootBrandSurface
[x] splash Expo : image = lirie-logo-color.png, imageWidth 220, fond #EAF3F1
[x] BootBrandMark JS aligné 220×95 (même wordmark, pas de saut de taille)
```

Rebuild obligatoire : Fast Refresh / OTA ne changent pas le splash Android. Profil cible : `eas build -p android --profile preview` (APK interne, **pas** `developmentClient`).

✅ **Implémenté** (avant smoke) : `hideNativeSplashWhenReady.ts`, `BootSplashGate` `onLayout` → `hideAsync`. `useBootSplashGate` : skip Lottie ≠ skip overlay.

✅ **Implémenté** (correctif post-smoke) : constante unique `SPLASH_BACKGROUND_COLOR` dans `src/core/boot/bootSurface.ts`. `BootBrandMark` / `BootBrandSurface` (wordmark `lirie-logo-color.png` RGBA). Overlay natif + web : logo permanent. `app/_layout.tsx` : plus de root blanc, fonds `#EAF3F1`. `app/index.tsx` : hold et Redirect sur `BootBrandSurface`. `app.json` splash : logo + `#EAF3F1` (prend effet au prochain prebuild / build natif ; le logo JS passe par Metro tout de suite).

### Re-smoke DRIVER-COLD-01 (2026-09-06 16:16)

```text
adb connect 192.168.1.33:40639
SM_S911B = device
force-stop ch.liri.operations
pause 1 s
monkey LAUNCHER 1
aucun Fast Refresh / reload Metro
```

Captures : `docs/ops/_smoke_driver_cold_01_2026-09-06/`

```text
[ ] fond #EAF3F1 continu          FAIL — blanc ~1,9 s → ~12 s
[ ] logo/branding visible/stable  FAIL — natif sans logo ; blanc sans logo ;
                                  logo OK seulement à t+15 s (overlay JS)
[ ] aucun flash blanc             FAIL
[ ] aucun menthe → blanc → menthe FAIL
[x] overlay jusqu'au shell prêt   overlay brandé visible à t+15 s
[x] hub peint ensuite             t+18 s (sheet flotte = hors critère boot)
```

Timeline :

```text
t+0,8–1,4 s   menthe plein écran, SANS logo     splash natif APK (splash-solid)
t+1,9–5,5 s   BLANC plein écran                 ← flash
t+7–9 s       BLANC + « Loading from 127.0.0.1:8081… »
t+12 s        BLANC                             bandeau Metro disparu
t+15 s        menthe + logo Lirie               overlay JS (correctif tient)
t+18 s        hub chauffeur peint
```

Moment du blanc : **natif → overlay**. Ce n’est pas overlay → hub. Le Dev Client / Metro peint un root `#FFFFFF` **avant** que le JS puisse monter `BootSplashGate`. Metro lent n’est pas le FAIL ; le FAIL est que cette attente Metro est blanche et sans logo, au lieu de menthe + branding.

Lecture figée après re-smoke :

```text
0,8 s      splash natif APK (menthe, sans logo)     ← FAIL natif (splash-solid)
1,9–12 s   Dev Client / Metro blanc                 ← DEV ARTIFACT, pas un défaut produit
15 s       overlay JS Lirie                         ← PASS
18 s       hub                                      ← PASS
```

`BootBrandSurface` / React n’existent pas encore pendant Metro. Ne pas chercher à peindre ces 10 s en JS.

✅ **Implémenté** (splash natif, rebuild pending) : plugin `expo-splash-screen` = `#EAF3F1` + `lirie-logo-color.png` + `imageWidth` 220 (bloc android identique). `BOOT_BRAND_LOGO_WIDTH/HEIGHT` = 220×95 pour coller au splash natif. `check-build-ready.js` refuse un retour à `splash-solid`.

**Reste** : commit des changements splash → `eas build -p android --profile preview` → installer l’APK sur S23 → **DRIVER-COLD-01 PRODUCT SMOKE** (force-stop → relaunch, sans Metro).

### Smoke visuel bundle Metro **froid** — 2026-09-06 20:45 (S23, même Dev Client)

Condition fautive reproduite (pas le bundle chaud 210 ms / 1 module) :

```text
Metro --clear, cache vide
Android Bundled 39859ms index.js (3095 modules)
force-stop → 1 s → monkey
vidéo 110 s depuis avant le lancement
aucun Fast Refresh / aucune modif GPS-queue
```

Vidéo : `docs/ops/_smoke_driver_cold_01_visual_2026-09-06/DRIVER-COLD-01-VISUAL.mp4`

```text
vidéo 0–14 s    écran précédent (preuve pré-lancement)
~15 s           menthe #EAF3F1 PLEIN ÉCRAN, SANS logo     splash natif
16–62 s         BLANC + barre Metro
                « Loading from 127.0.0.1:8081… »
                puis « Bundling 80%… 97%… »
66–70 s         hub + feuille « Disponibilité flotte »
```

JS (après le bundle, pas une preuve visuelle) : `BootSplash ready` 4 658 ms, overlay retiré 5 155 ms. Ignorés pour le verdict.

| Critère | Verdict |
|---|---|
| fond `#EAF3F1` continu | **FAIL** — blanc ~45 s |
| logo stable | **FAIL** — splash sans logo ; blanc sans logo |
| aucun blanc | **FAIL** |
| aucun menthe → blanc → menthe | **FAIL** (menthe → blanc → hub) |
| overlay jusqu’au shell, hub une fois | overlay JS non observable sur les frames ; hub ensuite |

```text
DRIVER-COLD-01      = FAIL / VISUAL CONFIRMED (bundle froid)
DRIVER-QUEUE-409-01 = RESTE OPEN
DEPLOY              = RESTE BLOQUÉ
```

Le blanc Metro est un **DEV ARTIFACT**. Fermeture produit = APK preview / standalone, **sans** Metro.

## DRIVER-COLD-02 — Stable dashboard shell

**Statut** : **PASS / CLOSED** — le smoke a confirmé header / disponibilité / mission stables une fois le hub peint. Ne pas rouvrir.

```text
[x] slot mission minHeight réservé
[x] skeleton = géométrie carte mission (pas 3 cartes)
[x] idle remplit le même slot
[x] slot carte hauteur fixe + placeholder (plus de rectangle vide)
[x] availability inconnue = bone, jamais En/Hors service inventé
[x] nom local / bootstrap immédiat, sinon bone ; avatar 46×46 fixe
```

✅ **Implémenté** : `driverDashboardShell.ts`, `DashboardMissionSlotSkeleton`, `DashboardMapPlaceholder`, hub `app/(app)/(driver)/index.tsx`, header `DriverDashboardHeader`.

## DRIVER-COLD-03 — ASYNC STATUS ≠ LAYOUT SHIFT

**Statut** : **IMPLEMENTED / DEVICE VALIDATION PENDING** — pas classé FAIL.

Le chip « Suivi à vérifier » à t+38 s dans une zone déjà à 48 px est le comportement voulu **si** `mission.y` ne bouge pas.

```text
Y mission avant chip == Y mission après chip  →  03 = PASS
```

Log DEV temporaire `onLayout` :

```text
[driver-shell-layout] statusArea height=48 y=... mode=empty|single|summary
[driver-shell-layout] mission y=...
```

✅ **Implémenté** : `driverStatusArea.ts`, `useDriverStatusIssues.ts`, `DriverStatusArea.tsx` (slot 48 + modal tracking), hub. `DriverTrackingBannerHost` reste `position: absolute`. `DriverStateBanners` conservé hors hub.

✅ **Implémenté** (validation) : logs `onLayout` DEV sur `DriverStatusArea` et le slot mission du hub.

✅ **Implémenté** (espacement vertical hub) : `DRIVER_DASHBOARD_HEADER_TO_STATUS_GAP = 0`, `DRIVER_DASHBOARD_STATUS_TO_MISSION_GAP = 14`. Plus de `gap: 10` global sur la page (il doublait le vide autour de la zone 48). StatusArea **reste 48 px**. Vide header→mission vide ≈ 62 px (48+14), pas 150. Header `alignItems: flex-start` pour éviter un centrage qui gonfle le bloc. Aucun changement GPS.

**Verdict mesure relative (`onLayout`)** : `status y=56` + `height=48` + `mission y=128` → GAP 24 px dans ce référentiel, visuel beaucoup plus large. Les `onLayout` ne partagent pas le référentiel écran du header. **FAIL** — on arrête de rogner des `gap`.

✅ **Implémenté** (StatusArea dans le header) : les 48 px DRIVER-COLD-03 restent, mais **dans l’enveloppe fixe du header** (plus 48 px de flux entre header et mission). Chip « Suivi à vérifier » = mêmes 48 px internes → `missionTop` inchangé. Flux header→mission = 16 px. Mesure DEV : `measureInWindow()` → `headerBottom`, `statusTop`, `statusBottom`, `missionTop`, `GAP_A`, `GAP_B`, `VISIBLE_GAP`.

✅ **Implémenté** (FAIL visuel 48 px) : plus aucune réserve StatusArea. Ligne 3 du header = `Localisation en cours…` **ou** `Suivi à vérifier · Vérifier`. `DRIVER_DASHBOARD_STATUS_TO_MISSION_GAP = 24`. Objectif device : `statusBottom → missionTop = 24–32 px`. MissionCard / GPS / FGS inchangés.

✅ **Mesure device** 20:05 :

```text
statusArea height = 12   (plus 48)
GAP_B             = 24   (statusBottom → missionTop)
VISIBLE_GAP       = 24
mode              = single  (alerte sur la ligne 3)
missionTop        = 84   (était 132)
```

Gate numérique 24–32 px : PASS. Screenshot overlay « keep awake » a masqué le hub — densité visuelle à confirmer à l’œil.

## DRIVER-COLD-04 — 0 ou 1 recentrage caméra

**Statut** : **IMPLEMENTED / BLOCKED BY MAP** — ce n’est pas un FAIL caméra.

Le smoke n’a jamais atteint les préconditions :

```text
pas de tuiles
pas de marker
pas de premier fix observable
```

Ne pas retoucher `driverMapCameraPolicy.ts` tant que la carte DEV est beige.

✅ **Implémenté** : `driverMapDisplayPosition.ts`, `driverMapCameraPolicy.ts`, `driverMapViewportStore.ts`, `MissionMap.tsx`. Badge LIVE seulement si `source === "gnss"`. Aucun changement `recorded_at` / watch / FGS / self-heal / cadence.

### Diagnostic Maps DEV (logcat S23, 2026-09-06) — sans toucher au GPS

```text
A. Maps ne charge pas ses tuiles     → Maps / dev build / API key / network
B. Maps charge mais aucun GNSS       → observation du premier fix
C. Maps + GNSS OK                    → alors seulement tester la caméra
```

Logcat (PID app, ~16:11) :

```text
MapsInitializer preferredRenderer: LATEST
Google Play services package version: 263332035   ← Play Services OK
Google Android Maps SDK renderer maps_core        ← SDK initialisé
AUCUN « Authorization failure » Android Maps SDK dans ce buffer
MapViewDirections:
  This IP, site or mobile application is not authorized to use this API key
  Request received from IP address 194.230.196.30, with empty referer
```

Lecture :

- Ce n’est **pas** un défaut Play Services / `MapsInitializer`.
- Pas de preuve d’`Authorization failure` sur le SDK tuiles dans ce buffer.
- L’erreur visible est **Directions HTTP** (`MapViewDirections`) : la clé native (`resolveGoogleMapsNativeApiKey`) est envoyée en requête web (referer vide). Une clé restreinte Android est refusée en HTTP — ça casse l’itinéraire, **pas forcément** les tuiles.
- Le beige observé peut être le fond style LIRIE (`landscape.natural` `#e4ebe7`) si les tuiles n’arrivent pas, ou un canvas style sans routes visibles.
- `eas.json` profil `development` n’embarque aucune clé Maps (secrets EAS uniquement).

**Ne pas modifier le GPS** pour une carte beige. **Ne pas** ouvrir 04 device validation avant tuiles + marker + premier fix observables.

## P0 GPS loop (distinct de l’écran blanc)

✅ **Implémenté** + **device PASS** (20:29, 60 s immobile) : AppState ignoré → 0 start/stop ; 1 stop tâche précédente + 1 start `mission_started` owner 45711 ; `resume_epoch = 0`. Voir `docs/ops/driver-runtime-01-refresh-storm-2026-09-06.md`.

`DRIVER-COLD-01` reste FAIL / VISUAL PENDING. Le smoke visuel natif → overlay → hub est le seul capable de le fermer.

## Ordre de travail

```text
P0     BOUCLE GPS AppState  CLOSED
P0     DRIVER-RUNTIME-01    PASS
       → smoke visuel 01 (natif → overlay → hub) pour fermer DRIVER-COLD-01
P0     DRIVER-RUNTIME-01B  barrière réseau / epoch 0 / FCM single-flight / HOLD PRESENCE / coalesce
       → même force-stop → relaunch ; ne pas fermer 01 tant que 401 pré-READY ou faux epoch
PUIS   HUB SPACING        STRUCTURAL PASS ; VISUAL DENSITY à confirmer (headerContentBottom)
ENS    PRODUCT SMOKE 01   APK preview sans Metro
02     CLOSED
03     48 px tenus, désormais internes au header
04     BLOCKED BY MAP CONFIG — ne pas toucher la politique caméra
```
