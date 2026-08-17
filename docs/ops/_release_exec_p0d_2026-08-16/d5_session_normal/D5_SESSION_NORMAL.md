# P0-D D5 — Session normale Prod126 (sans force-stop)

```text
FORCE-STOP TEST          = ABANDONNÉ pour D5 ✅
PROD126 INITIAL DELIVERY = PASS ✅
INITIAL 1/1              = PASS ✅ / CONFIRMÉ ✅
FIRST 1→2                = OBSERVÉ (explosion 1→42) ✅
D5 CAUSE FINALE          = EN ATTENTE (delivery post-storm)
PATCH                    = NO-GO
DISTRIBUTION             = NO-GO
```

## Flip observé (FG1, usage normal)

| Temps | fg / binds | Delivery |
|-------|------------|----------|
| FG1 +0…+283 s | **1 / 1** | PUT/LOC >0, Finished sporadique |
| FG1 +300 s ~21:19:48 | **42 / 43** | PUT=21 LOC=21 encore |
| HOME1 +0 s ~21:20:07 | **58 / 59** | PUT=12 LOC=12 |

`TRIGGER_211948_svc.txt` : `startForegroundCount=58`, Delivered Starts **#1 id=2 à −50.9 s** puis ~1 start / **300 ms** jusqu’à id=59 ; `startRequested=false` ; `getFgsAllow*=DENIED`.

```text
Fenêtre critique ≈ 21:18:57 → 21:19:48 (+02)
(premier extra startService → dump TRIGGER)
```

À ce stade : **storm pendant que delivery résiduelle encore vivante** → penche **A (storm avant mort delivery)** — à confirmer si PUT/LOC tombent à 0 ensuite.

Logcat ±20 s du premier flip : **presque vide** (filtre release / buffer). Capture continue démarrée pour la suite : `logcat_continuous_post_flip.txt` (sans changer le protocole FG/HOME).

## Verdicts

```text
A) 1→N avant perte delivery     ← candidat après ce flip
B) delivery meurt à 1/1 stable  ← infirmé pour CE run (storm avant mort)
C) 20 min sans panne            ← infirmé (storm en FG1 ~5 min)
```

## Protocole (inchangé)

FG 5 → HOME 5 → FG 5 → HOME 5, poll 30 s, **pas de force-stop**.

## Statut

```text
⏳ run continue (HOME1…) — surveiller mort delivery post-storm
```
