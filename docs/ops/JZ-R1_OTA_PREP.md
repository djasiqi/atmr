# JZ-R1 — Préparation OTA Android (instrumentation-only)

**Statut :** PRÊT À PUBLIER après PASS `JZ-R1_PREPROD_GATE.md` + **backend prod déployé**  
**Canal cible prod :** `production`  
**Flag par défaut bundle prod :** `EXPO_PUBLIC_ENABLE_TRACKING_PIPELINE_REMOTE_OBS` **absent (= OFF)**

---

## Prérequis ordonnés

```text
1. PREPROD GATE PASS
2. BACKEND PROD (migration 14d1b170291f)
3. SM-S911B canary (flag ON)
4. eas update production (Android)
5. Preuve distante Jozsef
```

Ne pas publier l'OTA avant le backend prod — un client instrumenté ne doit pas envoyer un payload que la prod ne persiste pas.

---

## Profils EAS

| Profil | Flag pipeline | Usage |
|--------|---------------|-------|
| `production` | OFF (défaut) | Rollout store / OTA général |
| `production-apk` | Ajouter `EXPO_PUBLIC_ENABLE_TRACKING_PIPELINE_REMOTE_OBS=1` pour lab | SM-S911B canary |
| `staging-canary` | Idem pour dev client | Tests internes |

Pour activer le canary Samsung **sans** activer tous les chauffeurs :

```json
"EXPO_PUBLIC_ENABLE_TRACKING_PIPELINE_REMOTE_OBS": "1"
```

Uniquement dans le profil de build/update canary, **pas** dans `production` tant que Jozsef n'est pas validé.

---

## Commandes (depuis `mobile/unified-app`)

### Preflight CI local

```powershell
npm run test:gps-critical
npm run typecheck:tracking
```

### Update OTA prod (Android) — **après GO explicite**

```powershell
cd mobile/unified-app
eas update --channel production --platform android --message "JZ-R1 tracking pipeline observability (instrumentation-only, flag OFF default)"
```

Capturer dans le gate :

- `updateId` / group ID Expo
- `runtimeVersion` aligné store
- SHA git des commits `feat(backend)` + `feat(mobile)`

---

## CI GitHub

Aucun workflow `eas update` automatique sur push — publication **manuelle** via EAS CLI (aligné historique GPS OTA).

Le push déclenche :

- `Phase 1 Gates` → `mobile-tracking-critical` inclut désormais `trackingPipelineObservability|trackingPipelineAnomaly`
- `deploy.yml` → backend image (sur merge main) — **ne pas confondre avec GO prod manuel**

---

## Rollback OTA

1. Republier un update précédent (Expo dashboard / `eas update:rollback` selon politique équipe)
2. Ou OTA avec `EXPO_PUBLIC_ENABLE_TRACKING_PIPELINE_REMOTE_OBS=0` explicite

Le backend reste backward-compatible : anciens clients sans `tracking_pipeline` continuent en 2xx.

---

## Preuve post-OTA (Jozsef)

Voir critères dans [`JZ-R1_PREPROD_GATE.md`](./JZ-R1_PREPROD_GATE.md).

Pour activer Jozsef après canary Samsung :

- Option A : OTA prod avec bootstrap `tracking_pipeline_remote_observability_enabled: true` (global — affecte tous les chauffeurs connectés)
- Option B : OTA dédié / channel canary si infrastructure Expo le permet
- Option C : attendre feature flag ciblable (non disponible aujourd'hui)

**Recommandation :** canary SM-S911B → bootstrap global ON pour cohorte pilote GPS (5 chauffeurs) → observation Jozsef.
