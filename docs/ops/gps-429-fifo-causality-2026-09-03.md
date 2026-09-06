# 429 ↔ FIFO iOS — causalité 3 septembre 11h–14h Zurich

**Date** : 2026-09-05  
**Fenêtre** : 2026-09-03 09:00–12:00 UTC (11:00–14:00 ZH)  
**Sources** : DLE (succès PUT, lag, seq) · Prometheus `http_requests_total` (429/200 flotte, pas de `driver_id`) · code limiter  
**Logs 429 par chauffeur** : indisponibles (backend redémarré le 5 sept)  
**Code GPS** : non modifié · plafonds non changés

## Phrase certaine

> **Les 429 PUT `driver_location` ne causent pas l’accumulation FIFO iOS.**  
> La file grossit alors que le débit de succès reste stable à ~1 Hz. Les 429 flotte sont un filet (0–8/min). Ils sont la **conséquence d’un frôlement de la fenêtre courte** (retry / `recorded_at` dupliqué), pas l’origine du retard 60–120 s.  
> **La clé du limiter est par `driver_id`**, pas par IP.

```text
CAUSE FIFO              capture ~1 Hz + upload utile légèrement inférieur
                        (plus quelques recorded_at répétés)
429                     conséquence ponctuelle (fenêtre short 30/10 s)
429 → FIFO              NON
FIFO → burst → 429      OUI, faible, en bord de short window
CLÉ LIMITER             http_rate:v2:driver_location:{short|long}:driver:{id}
CLÉ PARTAGÉE IP         NON
PLAFONDS                30/10 s et 120/60 s (env non surchargé)
AUGMENTER LES SEUILS    NON
```

---

## P0 — clé du limiter

Code (`driver_location_http.py`), défauts prod (aucune var `HTTP_DRIVER_LOCATION_*` dans le conteneur) :

```text
short : http_rate:v2:driver_location:short:driver:{driver_id}   30 / 10 s
long  : http_rate:v2:driver_location:long:driver:{driver_id}    120 / 60 s
```

- Flask limiter global : **exempt** sur PUT `/me/location`.
- Lua : le ZSET n’incrémente **que** les requêtes autorisées (un 429 ne « consomme » pas le quota).
- **D (clé partagée multi-chauffeurs) est éliminé**, sauf deux devices sur le même `driver_id`.

Un producteur propre à 1 req/s reste sous 3 req/s (short) et 2 req/s (long). Les 429 exigent un **surplus** : retry, même `recorded_at` renvoyé, ou 2ᵉ producteur.

---

## Débit succès vs plafonds (DLE = 1 PUT single)

PUT `/me/location` = **un point par requête**. Pas de batch dans cette voie (`/me/locations/batch` est un autre endpoint / un autre 429).

Fenêtre 09:00–12:00 UTC, rolling sur `created_at` :

| driver | max / 60 s | limite long 120 | max / 10 s | limite short 30 | dépassements |
| -----: | ---------: | --------------- | ---------: | --------------- | ------------ |
| 4 | **74** | sous | **31** | +1 | 1 fenêtre short |
| 7755 | **99** | sous | **31** | +1 | 2 fenêtres short |

Ils **n’atteignent jamais** 120/60 s en succès. Ils **effleurent** 30/10 s. C’est exactement là qu’un retry ou un PUT en trop produit un 429.

`sequence_id` dupliqué : **0**.  
`recorded_at` répété DRIVER-4 : **95 extras / 2 184** (~4 %). Pas un double 1 Hz, assez pour pousser 30 → 31 / 10 s.

---

## Test discriminant

### 7755 — la fenêtre propre (FIFO naît sous nos yeux)

11:56–12:32 ZH (09:56–10:32 UTC), ~60 succès/min **stables**, lag qui **monte tout seul** :

| min UTC | ok/min | lag moy | lecture |
| ------- | -----: | ------: | ------- |
| 09:56 | 31 | **2,8 s** | sain |
| 09:58 | 43 | **0,2 s** | sain |
| 10:00 | 60 | 1,5 s | 1 Hz, file vide |
| 10:05 | 60 | 8,7 s | file commence |
| 10:10 | 57 | 20 s | — |
| 10:15 | 57 | 30 s | — |
| 10:20 | 58 | 58 s | — |
| 10:21 | 58 | **62 s** | passe le seuil runbook |
| 10:25 | 59 | 85 s | — |
| 10:29 | 46 | **104 s** | plafond en vue |

Le succès **ne chute pas**. Le lag passe de 0 à 100 s en ~30 min.  
Si les 429 causaient la file (hypothèse **A**), le débit succès devrait s’effondrer **avant** la montée du lag. Ce n’est pas le cas.

Même motif recopié à 13:36 ZH (11:36 UTC) : lag 0,8 s → 71 s à 11:59, toujours ~58–60 ok/min. L’heure « 100 % tardive » 14h ZH est la **suite** de cette rampe.

### DRIVER-4 — 12h ZH (10:00 UTC) : file déjà pleine

À 09:18 UTC (11:18 ZH) : **3** succès, lag déjà **120 s**.  
À 09:31 : drain 42–60/min, lag 64–110 s.  
À 10:00–10:13 (12h ZH) : 59–60/min, lag 79 → 108 s.

On ne peut pas voir « 429 puis file » à 12h : la FIFO existe **avant**. Le 12h ZH montre seulement un **équilibre** upload ≈ capture (~1 Hz) qui **maintient** une file de ~90–110 s (elle ne se vide pas).

### 429 flotte (Prometheus, sans driver_id)

Sur 11:00–14:00 ZH : **0 à 8** 429/min, typiquement 0–4. Pic horaire ~190 = ~3/min.  
Succès flotte : 70–150/min (4 + 7755 à ~60 chacun + le reste).

Un filet de 3 429/min sur ~120 PUT/min ne peut pas créer une rampe de lag de 30 min à succès constant.

Logs `HTTP driver location rate limit driver_id=` : **absents** (restart). Attribution chauffeur des 429 : non prouvable à la requête près. Inutile pour la causalité FIFO : le signal succès+lag suffit.

`sent_at` / `retry_count` / `queue_depth` : **non persistés** en DLE.

---

## Hypothèses A–D

| | Verdict | Preuve |
| --- | --- | --- |
| **A** 429 → retries → débit ↓ → FIFO | **NON** | 7755 : succès plat, lag ↑ |
| **B** FIFO d’abord → vidange / extras → 429 | **OUI (faible)** | file d’abord ; short window touchée (31/30) |
| **C** double producteur | **partiel** | 0 seq dup ; ~4 % `recorded_at` répétés ; pas un 2e 1 Hz |
| **D** clé partagée IP/company | **NON** | clé `…:driver:{id}` |

Mécanique FIFO la plus simple : capture ~1,0–1,05 Hz, upload utile ~1,0 Hz (60/min). Excédent ~3 points/min → ~90 s de file en 30 min. Les pauses d’upload (déjà vues : +38 s de lag d’un coup) accélèrent. Le plafond 121 s est `too_old_for_mode`, pas le limiter.

---

## Ce qu’on ne fait pas

- Pas d’augmentation des plafonds 30/10 ni 120/60.
- Pas de hotfix GPS.
- Pas de chantier batterie **dans cette passe**. Baseline extraite : [`gps-mobile-battery-baseline-2026-09-05.md`](gps-mobile-battery-baseline-2026-09-05.md).

**Diagnostic GPS hebdomadaire figé** (2026-09-05) : soak AMBER, moteur GPS clos. Voir [`gps-prod-soak-7d-2026-09-05.md`](gps-prod-soak-7d-2026-09-05.md).

---

## Statut GPS après cette passe

```text
GPS PIPELINE BACKEND       GREEN
GPS REDIS / FANOUT         GREEN
GPS LIVE p99 CAUSE         IDENTIFIED (FIFO iOS recorded_at→HTTP)
iOS LOCAL QUEUE            AMBER (capture 1 Hz, upload ≤ capture)
HTTP 429                   EXPLAINED as edge of short window
429 ↔ FIFO CAUSALITY       429 does NOT cause FIFO
RATE-LIMIT KEY             CONFIRMED per driver_id

7514 MOBILE SILENCE        OPEN
39067 ZOMBIE LIVE          OPEN

GPS HOTFIX                 NO
GPS CODE                   FROZEN
```
