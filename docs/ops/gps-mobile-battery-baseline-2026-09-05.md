# Baseline batterie mobile iOS — entrée chantier énergie

**Date** : 2026-09-05  
**Origine** : soak GPS prod 7 j (29 août → 4 sept) — diagnostic moteur **figé**  
**Code GPS** : gelé. **Aucune** modification de fréquence, rate limiter, Redis, fanout, Kafka dans ce document.

Ce fichier est le **point de départ** de la discussion **GPS MOBILE BATTERY / ENERGY OPTIMIZATION**.  
Le soak 7 j est **clos**. Ne pas remettre en cause le diagnostic pipeline / p99 / 429.

## Règle du chantier

**Mesurer avant de modifier.**

Puis chercher des gains sur : localisation, wakeups, FGS, réseau, batching / file locale, précision, comportement PRESENCE / LIVE.

Surveiller **systématiquement** après toute expérience :

- p95 / p99 LIVE
- continuité (trous intra-mission)
- faux LIVE (surtout après COMPLETED)

Une baisse de conso qui dégrade ces trois signaux est un échec.

## Objectif

Réduire **sensiblement** la consommation des chauffeurs (surtout iOS) tout en **conservant — voire en améliorant** — la qualité LIVE déjà mesurée.

Les items ci-dessous sont des **hypothèses à mesurer**, pas des correctifs immédiats.

## Ce qui est déjà figé (ne pas réenquêter)

| Fait | Preuve |
| ---- | ------ |
| Pipeline HTTP → Redis canonical → Socket.IO | GREEN, 0 ms de traitement sur le p99 |
| p99 LIVE 117 s | FIFO iOS `recorded_at → received_at` |
| Backend / Kafka / fanout | pas la cause |
| 429 → FIFO | **NON** |
| Clé rate-limit | `http_rate:v2:driver_location:{short\|long}:driver:{id}` |
| Plafonds | 30 / 10 s et 120 / 60 s — **ne pas monter** |
| DRIVER-3 Android 1.0.13 | p99 56,8 s — le phénomène file est **iOS** |

Docs : [`gps-prod-soak-7d-2026-09-05.md`](gps-prod-soak-7d-2026-09-05.md) · [`gps-p99-drivers-4-7755-2026-09-05.md`](gps-p99-drivers-4-7755-2026-09-05.md) · [`gps-429-fifo-causality-2026-09-03.md`](gps-429-fifo-causality-2026-09-03.md)

Hors scope énergie (chantiers séparés) :

```text
7514   MOBILE CONTINUITY / FGS / OS LIFECYCLE
39067  BOOKING + ASSIGNMENT + 0 faux mission_live après COMPLETED
```

## Baseline iOS observée (1.0.13)

Devices représentatifs : **iPhone XR** (DRIVER-4), **iPhone 12** (DRIVER-7755), app **1.0.13**.

```text
capture                    ~1 Hz (LIVE HTTP)
recorded_at répétés        ~4 % (DRIVER-4, 3 sept 11–14h ZH)
upload utile               ~1 Hz (≈ 57–60 PUT succès / min)
FIFO locale                peut monter vers 70–120 s
rattrapage                 ponctuel (pauses upload ~30–40 s)
429                        filet, bord fenêtre courte 30/10 s
plafond accept             121 s = too_old_for_mode
PUT                        1 point / 1 requête (pas de batch sur cette voie)
```

Mécanique FIFO la plus simple déjà mesurée : capture ~1,00–1,05 Hz, upload utile ~1,00 Hz → excédent ~3 pts/min → ~90 s de file en 30 min.

Android (DRIVER-3, même app) : p99 sous 60 s — contraste utile pour le chantier.

## Hypothèses énergétiques à mesurer

L’iPhone dépense-t-il inutilement de l’énergie à :

1. **produire** une localisation toutes les secondes ;
2. **garder** le GPS au niveau de précision maximal en permanence ;
3. **réveiller** le réseau pour envoyer pratiquement chaque point individuellement ;
4. **maintenir** simultanément capture + FIFO + upload ;
5. **produire** les ~4 % de `recorded_at` répétés ;
6. **rattraper** une file alors que la carte n’a probablement pas besoin de chaque point intermédiaire ;
7. **garder** des activités PRESENCE avec des caractéristiques proches du LIVE.

Aucune de ces questions n’autorise un changement de cadence « pour voir » sans mesure avant / après (batterie **et** p50/p99 LIVE).

## Contraintes héritées du soak

```text
NO GPS HOTFIX
NO RATE LIMIT CHANGE
NO FREQUENCY CHANGE sans mesure
NO KAFKA CHANGE
NO REDIS / FANOUT CHANGE
```

Une optimisation énergie qui **dégrade** le p50 sain (~1,7 s chez DRIVER-4 hors heures FIFO) ou qui crée des trous LIVE > 10 min est un échec.

## Instrumentation minimale (prochaine étape)

Champs seulement — pas un système de télémétrie lourd :

```text
platform · device_model · app_version
tracking_mode · provider · enqueue_source
callback_at · recorded_at · enqueue_at · upload_at
queue_depth · duplicate_timestamp
watch_active · native_task_active
```

Puis par heure : callbacks natifs / JS, enqueues, unique fixes, PUT, duplicates, queue p50/p95/max.  
Coupes : PRESENCE vs LIVE · iOS vs Android · FG vs BG.

## Ordre de chantier figé

```text
MEASURE CALLBACKS FIRST
THEN REMOVE REDUNDANT WORK
THEN CHEAPEN PRESENCE
THEN OPTIMIZE IOS LOCATION PROFILE
THEN NETWORK / BATCHING

LIVE CADENCE LAST
```

## Cartographie code (2026-09-05)

Lecture seule : [`gps-mobile-battery-code-map-2026-09-05.md`](gps-mobile-battery-code-map-2026-09-05.md).  
✅ **Implémenté** : compteurs opt-in `batteryEnergyCounters.ts` + hooks natif/watch/enqueue/PUT. Aucun changement de cadence, FGS, PRESENCE ou réseau.

Hotspots déjà visibles dans Git : task natif iOS `distanceInterval=0` + High (1 Hz probable), PRESENCE au tarif LIVE, double enqueue (task + tick 8 s), drain 60/min, pas de batch HTTP.

## Statut

Campagne T0 (debug ≠ release) : [`gps-mobile-battery-t0-campaign-2026-09-05.md`](gps-mobile-battery-t0-campaign-2026-09-05.md).  
Flag : `EXPO_PUBLIC_ENABLE_TRACKING_BATTERY_ENERGY_INSTR` (OFF store).

```text
GPS PROD SOAK 7D           AMBER · DIAGNOSTIC COMPLETE · ENGINE CLOSED
BATTERY CODE MAP           COMPLETE / READ-ONLY
CODE CHANGES               INSTRUMENTATION ONLY
PHASE 1 DEBUG              NOT PASS UNTIL VALIDATED ON DEVICE
T0 FIRST AFTER PASS        PRESENCE vs LIVE · iOS · écran éteint
LIVE / FGS / RATE LIMIT    FROZEN
```
