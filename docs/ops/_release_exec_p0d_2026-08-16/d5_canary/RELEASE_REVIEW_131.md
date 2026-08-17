# RELEASE REVIEW — Binaire 131 (D5)

```text
DATE              = 2026-08-17
REVIEWER          = agent (release review)
BINARY            = 1.0.11 / versionCode 131
EAS BUILD         = fba42011-32ce-4ef7-b26e-91d08c79e318
ARTIFACT          = https://expo.dev/artifacts/eas/cYdNK8rkhKGPlzOmGFCsMIWYvcUN0-bolVMyszFSAU8.apk
PROFILE           = production-apk
CHANNEL EAS       = production
DISTRIBUTION EAS  = INTERNAL
CANARY C1–C4      = VALIDATED ✅
GPS UI freshness  = HORS SCOPE (chantier séparé)
```

---

## 1. Gel du binaire 131

| Élément | Valeur | Statut |
|---------|--------|--------|
| Build ID | `fba42011-32ce-4ef7-b26e-91d08c79e318` | ✅ FINISHED |
| versionCode / Name | 131 / 1.0.11 | ✅ |
| runtimeVersion | 1.0.11 | ✅ |
| Profil | `production-apk` (extends `production`, APK, `autoIncrement: false`) | ✅ |
| Artifact URL | pin APK ci-dessus | ✅ |
| `gitCommitHash` EAS | `101992cce16176661cb1e6a4f08eb01fedb29d3e` | ⚠️ |
| Contenu D5 dans git à ce SHA | **NON** — D5 est dans le working tree uploadé, **non commité** | ❌ BLOCKER |
| Post-canary code drift | mtimes D5 (`ownerVersionMismatchPolicy`, `backgroundLocationTask`, `app.json`) **avant** `createdAt` build (04:09+02) ; pas de modif post-C4 sur ces fichiers | ✅ |
| Variables profil effectives | env `production` EAS + overrides `production-apk` dont **`EXPO_PUBLIC_TRACKING_QA_PANEL=1`** | ⚠️ BLOCKER flotte |

```text
GEL REPRODUCTIBLE PAR SHA GIT = NO ❌
GEL PAR BUILD_ID + ARTIFACT   = YES ✅ (APK exact canary)
CANDIDAT FLOTTE PLAY (AAB sans QA) = NO ❌ (profil / flags)
```

### Env production effectives (résumé)

- API / socket : `https://api.lirie.ch` (profil production)
- Tracking flags P0 : BG location, queue, self-heal watch, etc. = ON (cf. `eas.json` production)
- `EXPO_PUBLIC_TRACKING_QA_PANEL=1` **uniquement** via extends `production-apk` → injects canary C3/C4 **actifs**
- Profil `production` (store AAB) : **pas** de `TRACKING_QA_PANEL` dans eas.json

---

## 2. Revue diff D5 (périmètre)

### Delta mobile vs `HEAD` (101992cc) — fichiers tracking

```text
app.json                                          versionCode 130→131
app/(app)/(driver)/_layout.tsx                    +canary installs
hooks.ts                                          +transient mission loss
backgroundLocationTask.ts / .test.ts              ownership STOP, FIX W1
driverTrackingBridge.ts / .test.ts                requestTrackingStop / B2
trackingSelfHeal.ts / .test.ts                    L1 non destructif + probes
+ ownerVersionMismatchPolicy.ts (+test)           FIX W1 policy
+ trackingLifecycleOwner.ts (+arch test)          ownership
+ canaryD5*.ts                                    injects QA (gated)
≈ +936 / −65 lignes (tracked) + untracked D5
```

### Protections validées présentes

| Protection | Présente | Note |
|------------|----------|------|
| Ownership unique STOP | ✅ | `requestTrackingStop` + `shouldAbortNativeStop` |
| B2 via lifecycle owner | ✅ | bridge |
| Self-heal L1 non destructif | ✅ | C4_131 PASS |
| Transient React loss | ✅ | hooks + C3 PASS |
| owner_version_mismatch | ✅ | policy L1 / owned stop ; T12 arch |
| Instrumentation `[D5-NATIVE]` | ✅ | gated QA panel |

### Chemins destructifs restants (hors owner generation)

| Chemin | Via | Owner gen? | Verdict review |
|--------|-----|------------|----------------|
| `owner_version_mismatch` | `requestTrackingStop` | ✅ | FIXED (W1) |
| `requestTrackingStop` → `stopBackgroundLocationTask` | bridge | ✅ | OK |
| `context_upgrade_to_mission` | `stopNative…Safely` → `requestNativeStop` | ❌ gen | **Résiduel accepté sous monitoring** (pas le W1) |
| Presence stop versionné | `Safely` + version check | partiel | OK métier |
| Headless lease / stale owner | `stopBackgroundLocationTask` | interne task | OK métier |
| `Location.stopLocationUpdatesAsync` hors `backgroundLocationTask` | arch T8 | — | ✅ interdit |

```text
DIFF D5 = PÉRIMÈTRE OK ✅
BYPASS W1 = ABSENT ✅
RÉSIDU context_upgrade = DOCUMENTED (non-blocker canary ; watch en rollout)
```

### Hors périmètre (ne pas mélanger)

- Working tree **backend/** / scripts / docs ops hors D5 : **sales** → ne pas inclure dans le commit freeze mobile
- UI « GPS Non confirmé / Aucun GPS récent » : **chantier séparé**

---

## 3. Gate qualité / CI

| Gate | Résultat |
|------|----------|
| Tests D5 (policy + arch T9–T12 + ensureNative) | **PASS** (inclus dans suite ci-dessous) |
| Tests tracking (selfHeal, bridge, bgTask, policy, arch) | **65 PASS / 5 suites** ✅ |
| `tsc --noEmit` global | dette préexistante (hors scope B3) ⚠️ |
| dont fichiers D5 touchés | **0 erreur** ✅ (B3 FIXED 2026-08-17) |
| Backend API / migrations pour D5 | **0** (canary observation-only) ✅ |
| Crash / ANR canary C4 | **0** observés sur fenêtre ✅ |

```text
QUALITÉ RUNTIME CANARY = PASS ✅
QUALITÉ TYPECHECK D5   = PASS ✅ (B3 FIXED — union + source)
CI GLOBALE TSC         = FAIL (préexistant, hors train D5)
B3                     = FIXED ✅ — SHA FINAL pas encore créé
```

---

## 4. Rollback (pré-distribution)

| Item | Valeur |
|------|--------|
| Version précédente stable Play | **versionCode 126** (AAB `a2970b22-…`, cf. `gps-p0-release-execution-2026-08-16.md`) |
| Comment revenir | Play Console → halt staged rollout / promote **126** ; devices sideload → réinstall APK 126 |
| Backend | **pas de rollback mobile obligatoire** ; G3 `BACKEND-ONLY ROLLBACK SAFE` ; D5 mobile-only |
| Métriques déclencheurs rollback mobile D5 | Unregister inattendu ↑ ; Register↔Unregister storm ; FGS mort sous mission ; PUT/LOC drop ; crash/ANR spike ; `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED` |
| Qui décide | Owner produit / ops (humain) — **pas** auto-rollback scripté mobile |
| Interdit | OTA channel `production` pour patch inject JS ; force-push ; mix backend dirty tree |

Réf. backend G4 : `docs/ops/gps-p0-g4-rollback-2026-08-16.md` + `previous-release.json` (skew API).

```text
ROLLBACK MOBILE PLAN = PRÊT CONCEPTUELLEMENT ✅
RUNBOOK MOBILE D5 DÉDIÉ = À FORMALISER (1 page) avant GO flotte
```

---

## 5. Verdict GO / NO-GO distribution

```text
D5 RCA              = CLOSED ✅
PATCH               = VALIDATED ✅ (canary C1–C4)
CANARY              = VALIDATED ✅
RELEASE REVIEW      = FAIL ❌

→ NO-GO DISTRIBUTION ⛔
```

### Blockers (ordre)

1. **B1 — Freeze SHA** : le canary n’est pas ancré sur un commit git contenant le patch D5 (`gitCommitHash` EAS = docs-only `101992cc`). Impossible d’auditer / rebuilder à l’identique depuis git. → **NEXT = commit SHA FINAL S (D5 + B3)**
2. **B2 — Profil candidat** : l’APK 131 est `production-apk` avec **QA panel + injects canary ON**. Ce n’est pas le binaire store `production` (AAB, sans QA). Distribuer 131 = distribuer un binaire d’instrumentation.
3. **B3 — Typecheck D5** : ✅ **FIXED** (union event names + `source` self-heal). 65 tests PASS ; tsc D5-related = 0.

### Non-blockers / acceptés

- Résidu `context_upgrade_to_mission` (STOP via `requestNativeStop`, hors gen owner) — surveiller en rollout
- Dette `tsc` globale 285 — hors scope D5 pour GO canary ; ne bloque pas le *contenu* patch si B1–B3 traités
- UI GPS freshness — hors D5

---

## 6. Chemin vers GO (sans re-canary C1–C4 si binaire = même code)

```text
1. ✅ FIX B3 (télémétrie) — DONE
2. Commit freeze D5-only + B3 → SHA FINAL S (+ tag RC)
3. Build profil `production` AAB (QA OFF), versionCode nouveau
4. Vérifier artifact : gitCommitHash = S ; QA panel absent
5. Runbook rollback mobile → 126
6. Smoke RC court (FG→HOME) puis GO/NO-GO rollout
```

```text
MAINTENANT     = COMMIT SHA FINAL S (après B3)
INTERDIT       = nouveau patch GPS diagnostic ; re-run C1–C3 ; OTA prod inject ; distribuer APK 131 QA ON
GPS UI         = chantier séparé
```
