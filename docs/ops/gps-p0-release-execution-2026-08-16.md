# GO RELEASE EXECUTION — GPS P0 (2026-08-16)

```text
═══════════════════════════════════════════════════════════
GPS P0 / BACKEND / LEDGER / OPS
= CLOSED / PASS ✅

ANDROID/iOS PRODUCTION BINARY
= BUILD / DISTRIBUTION VALIDATION EN COURS
═══════════════════════════════════════════════════════════

RELEASE TIP         = 286737a2362eb1e38013c72d04be23fcd608210e
TAG RC              = gps-p0-rc-2026-08-15
DOCKER TAG          = sha-286737a2362e
BACKEND DIGEST      = sha256:02053f7361d2000a61a0e13a0151b95a9c65481326c20453075037bc9129cbe1
WS DIGEST           = sha256:4d5afbdb203e940672c431e678922e48c37b7e27390f3096abf1b1cbcb42d2d3
ALEMBIC / PURGE     = AUCUN
FANOUT HOLD         = CONSERVÉ (Created, ENABLED=false)
```

## Deux statuts figés (ne pas mélanger)

| Piste | Périmètre | Statut |
|-------|-----------|--------|
| **A — Ops P0** | Backend/runtime prod + ledger + HOLD + G5 + canary tip | **CLOSED / PASS ✅** |
| **B — Binaire mobile** | Artefacts EAS production + smoke hors Metro avant diffusion chauffeurs | **EN COURS** |

Le canary diurne a validé le **tip JS via Dev Client / Metro sur staging**, **pas** le binaire EAS production distribué. Cela **ne rouvre pas P0** : c’est uniquement le contrôle packaging/distribution mobile.

## Runs GitHub

| Étape | Run | Résultat |
|-------|-----|----------|
| Build + Deploy API | https://github.com/djasiqi/atmr/actions/runs/31914382468 | success |
| Kafka P0 ingest recreate | https://github.com/djasiqi/atmr/actions/runs/31915403334 | success (consumer only) |
| Outbox + fanout/dlq align | SSH manuel post-run (même digest/tag) | OK |

## Topologie post-release (live)

```text
API / celery / flower / beat / ws   sha-286737a2362e   Up healthy
consumer                            sha-286737a2362e   Up healthy
outbox-publisher                    sha-286737a2362e   Up healthy
fanout-1 / fanout-2                 sha-286737a2362e   Created + ENABLED=false
kafka-dlq-consumer                  sha-286737a2362e   Created
```

`previous-release.json` installé sur `/srv/atmr/releases/previous-release.json` (rollback G4 skewé).

## Séquence exécutée

1. ✅ T-30 baseline — `docs/ops/_release_exec_baseline_T30_2026-08-16/`
2. ✅ Vérif tip / G0–G5
3. ✅ Push `release/gps-p0-2026-08-15` @ `286737a2`
4. ✅ Tag `gps-p0-rc-2026-08-15`
5. ✅ Build immuable
6. ✅ Artefact ↔ SHA (tag + digests)
7. ✅ Deploy API / celery / ws
8. ✅ Smoke health JSON `healthy`
9. ✅ Consumer (workflow P0) + outbox (align manuel même SHA)
10. ✅ Fanout/dlq recreate même SHA **sans** start ; HOLD OK
11. ✅ T+5 monitoring — `docs/ops/_release_exec_T5_2026-08-16/`
12. ✅ T+30 monitoring — `docs/ops/_release_exec_T30_2026-08-16/`
13. ✅ T+2h rétrospectif (~04:45 Genève / 02:45 UTC) — `docs/ops/_release_exec_T2h_2026-08-16/`
14. ✅ Canary LOC diurne — `docs/ops/_release_exec_canary_diurne_2026-08-16/canary_report.json` VERT
15. ✅ CLOSED (sauf rollback si seuil futur)

## T+5 (2026-08-15T23:47Z)

| Check | Résultat |
|-------|----------|
| api/ws/consumer/outbox Up healthy | ✅ |
| fanout not running (HOLD) | ✅ up=0, ENABLED=false |
| crash-loop | ✅ RestartCount=0 |
| anti-skew GIT_SHA | ✅ `286737a2…` partout |
| LOC rates 5m | ~0 (nuit, cohérent baseline T-30) |
| dedup invalid/unproven/claim | 0 |

**Verdict T+5** : VERT — aucun seuil IMMEDIATE.

## T+30 (2026-08-16T00:23Z)

| Check | Résultat |
|-------|----------|
| stack Up healthy, Restart=0 | ✅ |
| fanout HOLD | ✅ |
| LOC ingested/persisted vs baseline | ✅ rates ~0 = baseline nuit (pas de collapse) |
| dedup orphan / unproven | ✅ 0 |
| canary FG/BG/lock LOC | ⏳ non joué (nuit) — à faire en fenêtre diurne |

**Verdict T+30** : VERT ops (santé + anti-skew + HOLD). Preuve LOC réelle / canary reportée au diurne.

## T+2h rétrospectif (fenêtre 02:15–03:15 UTC = ~04:15–05:15 Genève)

| Check | Résultat |
|-------|----------|
| Restart=0 depuis deploy (StartedAt inchangé) | ✅ |
| images `sha-286737a2362e` | ✅ |
| fanout up=0 + ENABLED=false | ✅ (range Prom min=max=0) |
| orphan claim Redis `atmr:driver_location:event:*` | ✅ 0 |
| duplicate final / dedup unproven | ✅ 0 |
| 5xx rate | ✅ 0 sur toute la fenêtre |
| PG LOC/health/native_start_error dans fenêtre | ✅ 0 (nuit, pas de traffic) |
| dérive queue/ledger | ✅ aucun signal (persist rate 0, claims 0) |

**Verdict T+2h** : VERT (stabilité / HOLD / anti-erreurs). Volume LOC diurne hors scope de cette fenêtre.

### Reconfirm live diurne (2026-08-16T09:55Z / ~11:55 Genève)

| Check | Résultat |
|-------|----------|
| Restart=0, SHA tip inchangé | ✅ |
| Fanout HOLD up=0 | ✅ |
| 5xx (rétro + last 2h) | ✅ 0 |
| Claims Redis | ✅ 0 |
| native_start_error | ✅ 0 |
| LOC last 2h | 1 (driver 3) — pas de collapse / pas de storm |

Evidence : `docs/ops/_release_exec_T2h_2026-08-16/t2h_live_2026-08-16_daytime.txt`  
**Verdict T+2h reconfirm** : VERT ✅

## Canary diurne (piste A — tip JS, pas binaire EAS)

```text
Préparé : docs/ops/_release_exec_canary_diurne_2026-08-16/
Exécuté  : 2026-08-16 ~11:57–12:06 Genève
Cible    : Lirie Dev Metro tip 286737a2 → staging :15100
           (PAS le build EAS production installé hors Metro)
Driver   : 19 / mission 27 IN_PROGRESS
Verdict  : VERT ✅
  P0-A overlap/ERR_FG/native_start_error = 0
  P0-B auth_not_usable = 0
  LEDGER gen_null/HOL logcat = 0 ; LOC 4→25+ (39 rows mission 27 pendant canary)
  OBS nfix frais vu ; pas de native_start_error PG
Rapport  : canary_report.json
```

```text
POST-DEPLOY OPS = VALIDATED / CLOSED ✅
  PROD SHA / T+5 / T+30 / T+2h / CANARY DIURNE tip = PASS
  Correctif P0 déployé et validé côté backend/runtime/pipeline.
```

## Builds mobiles production (piste B — enveloppe binaire)

```text
STATUT PISTE B         = BUILD / DISTRIBUTION VALIDATION EN COURS
SOURCE                 = worktree détaché @ 286737a2 (PAS feat/tracking-p0-p7-firewall)
TAG RC                 = gps-p0-rc-2026-08-15
PROFIL EAS             = production
CHANNEL                = production
version / runtimeVersion = 1.0.11 / 1.0.11
SDK                    = 54.0.0
Soumission stores      = NO-GO tant que smoke binaire non VERT
eas update production  = NO-GO tant que runtime/channel vérifiés + GO explicite
Preuve                 = docs/ops/_release_exec_mobile_builds_2026-08-16/builds.json
```

| Plateforme | Build ID | native build | Statut lancement | URL |
|------------|----------|--------------|------------------|-----|
| Android AAB | `a2970b22-6b5c-4390-a81b-74588e06b50b` | versionCode **126** | IN_PROGRESS | [EAS](https://expo.dev/accounts/drinjasiqi/projects/operations-app/builds/a2970b22-6b5c-4390-a81b-74588e06b50b) |
| iOS App Store | `6012a3af-2a00-4035-945e-21edd9c3a374` | buildNumber **70** | IN_PROGRESS | [EAS](https://expo.dev/accounts/drinjasiqi/projects/operations-app/builds/6012a3af-2a00-4035-945e-21edd9c3a374) |

`gitCommitHash` EAS confirmé = `286737a2362eb1e38013c72d04be23fcd608210e`.

### Smoke test court après EAS FINISHED (Android production — pas C3)

Checklist figée : `docs/ops/_release_exec_mobile_builds_2026-08-16/SMOKE_ANDROID_PRODUCTION.md`

```text
build production installé
→ lancement standalone (aucun Metro / DevLauncher)
→ login chauffeur
→ mission active
→ quelques LOC foreground
→ HOME/background
→ lock/unlock
→ LOC persistées côté prod
→ overlap START/STOP = 0
→ native_start_error = 0
→ auth_not_usable = 0
→ generation=null = 0
```

Si PASS → `ANDROID PRODUCTION BINARY = VALIDATED / READY FOR DISTRIBUTION ✅`  
Pas de rejeu C3 ni campagne P0. Diffusion stores/OTA seulement sur GO explicite.

## Notes

- Le workflow `deploy-kafka-p0.yml` recreate **uniquement** `tracking-kafka-consumer` (by design HOLD). Outbox + fanout/dlq ont été alignés ensuite en SSH ciblé (`--no-deps`, fanout `--no-start`).
- `TRACKING_PROCESSED_FANOUT_ENABLED=true` peut apparaître sur API/consumer ; le HOLD effectif est sur les conteneurs fanout (`false` + Created).
- Pas d’Alembic, pas de purge Redis/Kafka.

```text
✅ **Implémenté** : piste A CLOSED — tip unique ; HOLD ; T+5/T+30/T+2h VERT ; canary tip diurne VERT.
  Builds EAS Android+iOS production lancés depuis 286737a2 (sans submit / sans OTA) = piste B.
**Reste à faire (piste B seulement)** : EAS FINISHED → smoke binaire production (sans Metro) → GO diffusion si demandé.
  Ne rouvre pas GPS P0 ops.
```
