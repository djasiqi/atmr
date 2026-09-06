# Campagne batterie T0 — instrumentation minimale

**Date** : 2026-09-05  
**Soak GPS 7D** : **clos** — ne pas relancer.  
**Code GPS** : gelé (cadence LIVE, FGS, PRESENCE/LIVE, réseau, rate-limit).  
**Seule modification autorisée ici** : compteurs opt-in, 1 événement / minute.

Flag : `EXPO_PUBLIC_ENABLE_TRACKING_BATTERY_ENERGY_INSTR=1`  
(`tracking_battery_energy_instrumentation_enabled`) — **OFF** en prod store.

## Statut chantier

```text
GPS MOBILE BATTERY

CODE MAP                  COMPLETE
ROOT CAUSE HYPOTHESES     IDENTIFIED
CODE CHANGES              INSTRUMENTATION ONLY

PHASE 1 DEBUG
STATUS = NOT PASS UNTIL VALIDATED ON DEVICE
BUILD                  battery-phase1-ios (EAS, iPhone physique)

NO BATTERY CONCLUSION
NO CADENCE CHANGE
NO LOCATION PROFILE CHANGE
NO PRESENCE CHANGE
NO BATCHING CHANGE
NO FGS CHANGE

AFTER PHASE 1 PASS — T0 FIRST
iOS · same device · same release · same instrumentation
screen off · comparable duration
PRESENCE vs MISSION_LIVE
→ batterie + Energy Log
→ callbacks / uniques / enqueues / PUT / queue / freshness
```

## Deux builds, deux questions

```text
1. DEBUG / DEV
   → l’instrumentation affiche-t-elle les bons compteurs ?
   → AUCUNE conclusion mAh / % batterie

2. RELEASE / PROD-LIKE
   → T0 énergétique réel (même app que les chauffeurs + flag ON)
```

Un debug (Metro, logs, debugger) fausse la conso. Phase 1 = validation fonctionnelle seulement.

## Build iPhone Phase 1 DEBUG

Profil EAS `battery-phase1-ios` : hérite de `development` (`developmentClient`, iPhone physique, `simulator: false`). **Seule** différence fonctionnelle :

```text
EXPO_PUBLIC_ENABLE_TRACKING_BATTERY_ENERGY_INSTR=1
```

```text
IPHONE PHASE 1 DEBUG

developmentClient           YES
physical iPhone             YES
instrumentation battery     ON

GPS Accuracy                UNCHANGED
distanceInterval            UNCHANGED
LIVE cadence                UNCHANGED
PRESENCE                    UNCHANGED
FGS                         UNCHANGED
queue/drain                 UNCHANGED
rate limit                  UNCHANGED
Kafka                       UNCHANGED
```

Commande : `npm run build:battery-phase1:ios` (depuis `mobile/unified-app`).  
En non-interactif, EAS a bien chargé le flag + l’héritage `development`, mais a refusé le provisionnement ad-hoc : relancer **en interactif** sur la machine qui a déjà les credentials iOS internes.

### Protocole iPhone (10–15 min) — pas de lecture batterie

1. Installer le build interne, ouvrir le Dev Client, **connecter Metro** (les snapshots `tracking.battery.minute` sortent en `console.info`).
2. Se connecter comme chauffeur.
3. Démarrer une mission / `mission_live`.
4. Rouler ~10–15 min.
5. 3–5 min au premier plan, puis arrière-plan + écran éteint pour le reste.
6. Terminer normalement.
7. Filtrer les logs : `[driver-telemetry] tracking.battery.minute`.

**Ensuite on arrête.** Pas de PRESENCE vs LIVE, pas de `%/h`, pas de changement GPS. Phase 2 T0 Release seulement après PASS.

## Coût télémétrie

- Compteurs mémoire (increments).
- **Un** événement `tracking.battery.minute` par minute calendaire, pas par callback.
- Flag OFF = un booléen, zéro timer, zéro emit.

Champs par snapshot :

```text
platform · device_model · app_version
tracking_mode · app_state · provider
native_callbacks · js_callbacks
unique_fixes · duplicate_fixes
enqueues · put_success
enqueue_native · enqueue_bridge_tick · enqueue_bridge_fallback
queue_depth_enqueue_min / max / last / p50 / p95
queue_depth_drain_min / max / last / p50 / p95
same_recorded_at_reused · layers_not_collapsed
callback_to_enqueue_p50_ms
enqueue_to_upload_p50_ms
recorded_to_upload_p50_ms
native_task_active · js_watch_active
```

Tableau horaire dérivé : `*_per_min` = totaux de la fenêtre / minutes.

Couches : un même `recorded_at` peut être callback natif + watch JS + 2 enqueues.  
**PASS couches** si, quand `bridge_tick` réutilise un fix :

```text
unique_fixes   <  enqueues
native_callbacks ≠ enqueues
same_recorded_at_reused = true
```

`native_callbacks` peut égaler `unique_fixes` (un callback = un timestamp). Ce qu’il ne faut pas : fusionner enqueue et unique, ou callback et enqueue.

## Phase 1 — critères PASS (debug)

Aucune conclusion batterie. GPS inchangé.

```text
PHASE 1 DEBUG — PASS SI

iOS LIVE
- native callbacks ≈ 1 Hz confirmé
- js_watch_active visible au foreground
- native_task_active visible
- bridge_tick séparé du native_task
- duplicates recorded_at quantifiés
- enqueue_source exhaustif
- queue_depth_enqueue et queue_depth_drain (min/max/last) cohérents avec la FIFO
- PUT success/min cohérent avec les logs réseau

LATENCIES
- callback→enqueue mesurable
- enqueue→upload mesurable
- recorded→upload reproduit le lag connu

STATE
- PRESENCE et MISSION_LIVE correctement distingués
- FG/BG correctement distingués
- aucun événement minute quand flag OFF

SAFETY
- aucune modification de cadence
- aucune modification Accuracy/distance
- aucun changement FGS
- aucun changement rate-limit

COUCHES
- un fix n’est pas fusionné entre callback / unique / enqueue
```

Attendu **actuel** iOS `MISSION_LIVE` FG (à confirmer, pas à « corriger ») : ~1 Hz natif, watch JS au FG, tick 8 s visible, Android ~20 s.

## Phase 2 — T0 release / prod-like

**Uniquement après PHASE 1 PASS.** Build : **1.0.13 + instrumentation uniquement**.

Premier comparatif (le plus discriminant) — avant A–D complets si le temps manque, **celui-ci d’abord** :

```text
iOS
same device
same app/release
same instrumentation
screen off
comparable duration / environment

PRESENCE
vs
MISSION_LIVE
```

C’est ce test qui tranche si le premier chantier d’optimisation est vraiment **PRESENCE** (hypothèse code) ou si une autre source domine.

| Test | Mode | Durée | Contexte |
| ---- | ---- | ----- | -------- |
| A | PRESENCE | 60 min | écran surtout éteint |
| B | MISSION_LIVE | 60 min | véhicule, écran éteint |
| C | MISSION_LIVE | 60 min | application au premier plan |
| D | MISSION_LIVE | 30–60 min | immobilité |

Enregistrer en parallèle : `% batterie` début/fin, device, iOS, mode, compteurs, p95/p99 freshness, trous.  
iOS contrôlé : Energy Log / Xcode **en plus** du `%` — pas à la place.

Cible de démonstration :

```text
PRESENCE aujourd’hui
≈ coût GNSS LIVE
≠ besoin métier LIVE
```

Exemple de lecture (chiffres **après** T0, pas maintenant) :

```text
                   PRESENCE       LIVE
Battery / heure      X %           Y %
Native cb/min        X             Y
Unique fixes/min     X             Y
PUT/min              X             Y
```

## Phases 3–4

Optimisation ciblée **après** T0, puis même protocole en T1.  
Comparer : batterie, callbacks, réseau, p95/p99, continuité, faux LIVE.

```text
DO NOT
- relancer GPS PROD SOAK 7D
- changer Accuracy / distance / PRESENCE / batch / LIVE avant T0
- conclure une conso depuis un build debug
```
