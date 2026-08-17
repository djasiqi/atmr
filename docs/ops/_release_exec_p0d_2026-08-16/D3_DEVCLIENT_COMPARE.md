# P0-D D3 — Dev Client vs Prod autour de `Location unavailable`

```text
SOURCE     = ab_same_device/B_devclient (HOME 60s + LOCK 60s, 13:58:49→14:01:21)
PROD REF   = d3_native (clean HOME 90s) + A_prod126 (session déjà DENIED)
PATCH      = NO-GO
```

## Question discriminant

```text
Prod 126 :
  Location unavailable → FGS tombe / LOC meurt

Dev Client :
  Location unavailable ?
      ├─ NON → pourquoi la disponibilité diffère ?
      └─ OUI mais FGS reste → différence de lifecycle service
```

## Verdict

```text
Dev Client = OUI mais FGS reste → branche droite ✅
```

Le warning Expo **n’est pas exclusif au release**. Il apparaît aussi sur Dev Client, à cadence ~10 s, y compris pendant HOME/LOCK — **sans** collapse FGS OS et **sans** arrêt LOC.

| Signal (fenêtre HOME→LOCK) | Prod 126 (d3_native clean) | Dev Client 125 |
|----------------------------|----------------------------:|---------------:|
| `Location unavailable…` | oui (~HOME+8s, cadence ~10s) | **oui** (21 hits 13:58→14:01) |
| `isForeground` | reste true (90s) | **reste true** |
| `startRequested` | reste true (90s) | **reste true** |
| `getFgsAllowStart` | TOP (puis DENIED si session empoisonnée) | **TOP + SYSTEM_ALLOW_LISTED** |
| LOC backend pendant BG | **STOP** ~+40s HOME | **continue** (≥12 points) |
| `stopSelf` / AM Stopping LocationTask | non (clean) | non |

Artefacts extraits : `d3_devclient/unavailable_by_second.txt`, `unavailable_home_window.txt`.

### LOC Dev (extrait `staging_loc_health.txt`)

```text
13:58:55 … 13:59:10 … 13:59:29 … 13:59:48 (HOME)
14:00:12 … 14:00:28 … 14:00:47 … 14:01:06 (LOCK)
```

Health BG : `app=background fgs=True ntask=True` à 13:59:50.

---

## Relecture causale (alignée)

```text
Location unavailable  ≠  cause suffisante du DENIED
                      =  signal upstream commun Expo LocationTaskConsumer
                         (prod ET dev)

Prod release :
  warning + delivery LOC qui meurt
  (+ path empoisonné → startRequested=false / DENIED)
  → recovery JS bloquée (pas de re-start FGS depuis BG)

Dev Client :
  même warning
  mais FGS OS + allowlist + LOC tiennent
  → pas de DENIED, pas de mort fonctionnelle
```

Donc :

- **D3-A** : warning consumer = **commun** (pas discriminant release-only)
- **D3-C** : **leading prod** — service/shell vivant, delivery/LOC cassée
- **D3-D** : **partiellement confirmé** — ce n’est pas l’apparition du warning qui est release-only, c’est la **suite** (LOC morte / éventuel collapse allow → DENIED)
- **DENIED** = conséquence d’un FGS/delivery déjà dégradé + tentative de recréation BG, pas cause initiale

---

## Statut

```text
D1  essentially excluded
D2  RULED OUT ✅
D3  LEADING ▶
  D3-C  LEADING (prod)
  D3-A  contributif mais NON discriminant (aussi sur Dev)
  D3-B  not seen (clean 90s)
  D3-D  PARTIAL ✅ (conséquence release ≠ warning lui-même)

P0-D PATCH           = NO-GO
GENERAL DISTRIBUTION = NO-GO
```

## Suite (toujours read-only)

1. Sur Prod clean : prolonger au-delà de 90s / capturer le flip `isForeground/startRequested` **s’il** arrive, avec filtres AM (`stopForeground` / `stopSelf` / `Stopping service`).
2. Localiser le site exact du log dans `expo-location@19.0.8` (AAR / consumer).
3. Ne pas patcher ; ne pas whitelister batterie.
