# P0-D / D5 — RCA session normale (read-only)

> **Supersédé pour T_FAIL / chaîne** par `D5_RCA_SORTIE_OBLIGATOIRE.md`  
> (LAST GOOD = Finished 21:18:44.491 ; FIRST FAIL = Unregister 21:18:49.975 ;  
> JS STOP path @ 1er Unregister = EXCLUDED).  
> Ce fichier garde les timestamps poll/PG ; ne plus utiliser comme verdict B.

```text
NOM = RELEASE-ONLY EXPO LOCATION DELIVERY FAILURE
PATCH = NO-GO
DISTRIBUTION = NO-GO
BACKEND = GELÉ (lecture seule SSH)
```

## 1. VERDICT

```text
VERDICT = A (REBIND STORM LEADING / co-causal) = D5-A CONFIRMED ✅
```

Pas **B** (obsolète : T_FAIL est déterminé).  
Pas **C** (storm reproduit en FG1 ~5 min).

## 2–4. Timestamps

| | (+02) | (UTC) |
|--|-------|-------|
| **LAST KNOWN GOOD** | **21:18:51** LOC seq30 ; sample **21:18:59** encore `1/1` | 19:18:51 LOC |
| **FIRST FAILURE (HTTP/LOC)** | dernier PUT dense **21:18:54** ; gap LOC démarre | 19:18:54 / après 19:18:51 |
| **FIRST OBSERVABLE DIVERGENCE (AM)** | 1er Delivered Start extra **~21:18:57** (`id=2`, −50.9 s vs dump 21:19:48) | ~19:18:57 |
| **FIRST POLL DIVERGENCE** | **21:19:48** `1/1 → 42/43` | 19:19:48 |
| **PUT/LOC poll = 0** | **21:20:55** (HOME1+49s) | 19:20:55 |
| **LOC reprend (seq reset 1)** | **21:21:14** | 19:21:14 |

Gap PG : **143.4 s** entre `19:18:51` et `19:21:14` (mission **38224**, driver **20135**).

## 5. REBIND STORM

```text
REBIND STORM = CAUSAL / LEADING ★
(pas conséquence tardive d'une longue mort à 1/1)
```

Cascade ~**1 startService / 300 ms** → `startForegroundCount`/`binds` ×50–70 ; `startRequested=false` ; `getFgsAllow*=DENIED`.

## 6. CHAÎNE CAUSALE (observée)

```text
FG1 sain 1/1 + PUT/LOC vivants (~5 min)
  → T≈21:18:51–54  dernier LOC/PUT
  → T≈21:18:57      premier LocationTaskService start extra (id=2)
  → T≈21:18:57–21:19:48  storm startService/bind (~300 ms)
  → T=21:19:48      poll voit 42/43 ; counters PUT encore non nuls (fenêtre 90s)
  → T=21:20:07      HOME ; storm continue 58/59
  → T=21:20:55      PUT=0 LOC=0 poll ; storm 69/70
  → T=21:21:14      LOC reprend seq=1 (nouvelle génération native)
  → ensuite fg stabilise ~19/19 ; delivery intermittente / morte
```

Qui provoque chaque +1 : **non tranché au code** (logcat ±20 s du flip **vide** en release). Candidats code : `startLocationUpdatesAsync` / `fgs_recover` / watchdog (`backgroundLocationTask.ts`) — **hypothèse**, pas preuve log.

## 7. EVIDENCE TABLE (±30 s autour T_FAIL)

| t (+02) | FG/BG | fg/binds | Finished | PUT/LOC | Note |
|---------|-------|----------|----------|---------|------|
| 21:18:10 | FG | 1/1 | 0 | 22/22 | sain |
| 21:18:51 | FG | (1/1) | — | LOC seq30 | **dernier LOC** |
| 21:18:54 | FG | — | — | dernier PUT gw | **silence HTTP** |
| 21:18:57 | FG | 1→… | — | silence | **1er startService extra** |
| 21:18:59 | FG | **1/1** | 0 | 19/19* | *fenêtre 90s trompeuse |
| 21:19:48 | FG | **42/43** | 0 | 21/21* | TRIGGER_211948 |
| 21:20:07 | HOME | 58/59 | 0 | 12/12* | storm continue |
| 21:20:55 | HOME | 69/70 | 0 | **0/0** | delivery poll morte |
| 21:21:14 | HOME | — | — | LOC seq**1** | reset session |

## 8. EXCLUSIONS

- P0-A/B/C, D4-B comme cause de **ce** cut (backend reçoit encore 202 jusqu’au silence ; pas de conflit analysé ici)
- Request FLP params (A/B + dumpsys)
- Mock / stationnaire seul
- « Prod cassé dès le boot » (initial 1/1 + delivery PASS)
- Scénario C pour ce run
- IP `138.155.201.155` : timeout — host réel = `SERVER_HOST` local (`138.201.155.201`)

## 9. NEXT DISCRIMINANT (un seul, read-only)

```text
Logcat continu non filtré (ou tags Expo/AM) pendant FG jusqu'au prochain 1→N
+ corrélation telemetry fgs_recover / start_requested / watchdog
→ identifier QUI appelle startLocationUpdatesAsync / startService au premier +1
```

Sans ça : on sait **quand** et que le storm **mène** la panne, pas **qui** dans JS/natif.

## 10. PATCH

```text
PATCH = NO-GO
(premier étage « qui » pas encore démontré)
```

## Preuves

- `d5_session_normal/samples.csv`, `timeline.txt`, `TRIGGER_211948_*`
- SSH gateway PUT + PG `driver_location_events` driver 20135
- Git tracking : recovery FSM / `fgs_recover` présents (`backgroundLocationTask.ts`) — candidat seulement

## Sentry

Recherche MCP `lirie-mobile` : à relancer (query OR invalide). Ne bloque pas le verdict A basé sur device+SSH.
