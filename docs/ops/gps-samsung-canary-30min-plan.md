# Canary Samsung — run complet 30 min (C01→C11 + GPS)

Plan figé : **2026-08-23**.  
Objectif : **une seule session continue** sur **SM-S911B** — toutes les étapes passent d’affilée, GPS fonctionnel bout-en-bout.

Références :

- Matrice états : [`gps-driver-state-certification.md`](./gps-driver-state-certification.md)
- APK / Metro staging : [`gps-android-canary-apk.md`](./gps-android-canary-apk.md)
- Run précédent : [`_driver_state_cert_2026-08-21/C01_C10_RUN.md`](./_driver_state_cert_2026-08-21/C01_C10_RUN.md)
- Pilote prod 5 : [`gps-production-pilot-closed.md`](./gps-production-pilot-closed.md)

```text
VERDICT RUN
C01→C11 tous PASS + carte OK + invariants 0  → CANARY 30 MIN = PASS ✅
1 gate critique FAIL                         → STOP immédiat (pas C02…C11)
```

**Runs 2026-08-23** : **INVALID** — voir [`_samsung_canary_30min/SAMSUNG_CANARY_30_INVALID.md`](./_samsung_canary_30min/SAMSUNG_CANARY_30_INVALID.md). Ne pas agréger les runs chevauchés.

### Pré-gates obligatoires (script v2)

```text
G1  Aucun run_canary_30 / run_step actif (CANARY30.lock + scan PID)
G2  Foreground = ch.liri.operations/.MainActivity — sinon STOP (fermer DevLauncher)
G3  1 gate FAIL → FIRST_STOP — ne jamais continuer C02…C11
```

Implémentation : `_samsung_canary_30min/_canary_gates.ps1` · `run_canary_30.ps1` · `run_step.ps1`


## Périmètre

| Élément | Valeur |
|---------|--------|
| Device | Samsung **SM-S911B** (canary) |
| ADB | `adb-RFCW20QC53W-CDvueV._adb-tls-connect._tcp` (adapter si changé) |
| Compte | `atmr1@atmr.ch` / driver **20** |
| App | `ch.liri.operations` · dev client + **Metro** |
| BG Samsung | `adb shell input keyevent KEYCODE_HOME` ✅ |
| Durée cible | **30 min** (soaks courts — 2–3 cycles P9 par gate) |

**Hors scope** : deep Doze 60 min (B3 déjà CLOSED), flotte 2 devices, iOS.

---

## Chronologie (30 min)

| Min | Phase | ID | Action | Durée |
|-----|-------|-----|--------|-------|
| 0–3 | **P0** Preflight | — | ENV + device + permissions | 3 min |
| 3–5 | **P1** Smoke GPS | S0 | Login · 1er fix · projection | 2 min |
| 5–7.5 | **P2** | **C01** | FG · sans mission · PRESENCE ~60 s | 2.5 min |
| 7.5–10 | **P2** | **C02** | BG HOME · PRESENCE ~60 s | 2.5 min |
| 10–12 | **P3** | **C03** | Mission **ASSIGNED** · FG · LIVE ~20 s | 2 min |
| 12–14 | **P3** | **C04** | ASSIGNED · BG HOME · LIVE ~20 s | 2 min |
| 14–16 | **P4** | **C05** | **EN_ROUTE** · FG · LIVE | 2 min |
| 16–18 | **P4** | **C06** | EN_ROUTE · BG HOME | 2 min |
| 18–19.5 | **P5** | **C07** | **ARRIVED** · FG | 1.5 min |
| 19.5–21 | **P5** | **C08** | ARRIVED · BG | 1.5 min |
| 21–22.5 | **P6** | **C09** | **IN_PROGRESS** · FG | 1.5 min |
| 22.5–24.5 | **P6** | **C10** | IN_PROGRESS · BG ★ gate critique | 2 min |
| 24.5–27 | **P7** | **C11** | Terminale → soft PRESENCE 60 s | 2.5 min |
| 27–28 | **P8** | Carte | Dashboard entreprise 1/N live | 1 min |
| 28–30 | **P9** | Score | Markers + verdict | 2 min |

Si une étape **FAIL** → **STOP** (ne pas enchaîner) · RCA · artefact logcat conservé.

---

## P0 — Preflight (3 min) — **bloquant**

### Host

```powershell
cd c:\Users\jasiq\atmr
docker compose -f docker-compose.staging.yml --env-file .env.staging --profile canary up -d canary-gateway
curl.exe -sS http://127.0.0.1:15000/health
curl.exe -sS http://127.0.0.1:15100/health
curl.exe -sS http://127.0.0.1:8081/status   # packager-status:running
```

### Device Samsung

```powershell
$d = "adb-RFCW20QC53W-CDvueV._adb-tls-connect._tcp"
adb devices -l
adb -s $d reverse tcp:8081 tcp:8081
adb -s $d reverse tcp:15100 tcp:15100
adb -s $d shell pidof ch.liri.operations
```

### Téléphone (manuel)

- [ ] GPS ON · localisation **Toujours** pour Lirie
- [ ] Batterie Lirie : **non optimisée** / sans restriction
- [ ] Notifications Lirie autorisées (FGS)
- [ ] Chauffeur **EN SERVICE** (disponible)
- [ ] Pas de mission active au départ (ou script reset ci-dessous)
- [ ] Wi‑Fi stable · **pas** `api.lirie.ch`

### Reset mission driver 20 (optionnel, avant C01)

```powershell
docker exec atmrstg-backend-1 python /tmp/_samsung_canary_reset_mission.py
```

Script : [`_samsung_canary_30min/_samsung_canary_reset_mission.py`](./_samsung_canary_30min/_samsung_canary_reset_mission.py)

**Gate P0** : Metro UP · gateway UP · app pid OK · `age_s < 90` driver 20.

---

## P1 — Smoke GPS (2 min)

1. Ouvrir Lirie (FG) · confirmer connecté staging.
2. Attendre **2 fixes** (P9 ou `recorded_at` avance 2×).

```powershell
docker exec atmrstg-backend-1 python /tmp/_f03_ages.py
```

**PASS** :

- `mode=availability_presence` · `mission=None`
- `P8→J1→J7` au moins 1× dans logcat
- `driver.latitude/longitude` non nuls

---

## Grille commune (chaque C01→C11)

Pour **chaque** étape, avant de passer à la suivante :

```text
STATE / app_state (FG=MainActivity · BG=Launcher/HOME)
driver_id=20 · mission_id · mission_status
tracking_mode / task_mode
P8_count ≥ 2 · J1_count ≥ 2 · J7_count ≥ 2 · J7 sent≥1 backend_acked=1
P9_median ≈ 60 s (PRESENCE) ou ≈ 20 s (LIVE)
last_event_id · recorded_at avance
PG/DLE mode MATCH mission_id
Unregister = 0 · FLP_REMOVE = 0 · FGS_restart = 0
session stable (pas de rotate inattendu sauf TTL documenté)
VERDICT step = PASS / FAIL
```

### Soak court par étape

| Mode | Action | Timer | Min P9 |
|------|--------|-------|--------|
| FG | `am start -n ch.liri.operations/.MainActivity` | 90–150 s | 2 (PRESENCE) ou 3 (LIVE) |
| BG | `input keyevent KEYCODE_HOME` | idem | idem |

### Capture (chaque étape)

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$step = "C02"   # adapter
$d = "adb-RFCW20QC53W-CDvueV._adb-tls-connect._tcp"
$out = "docs\ops\_samsung_canary_30min"
adb -s $d logcat -c
# … soak …
adb -s $d logcat -d -v threadtime > "$out\logcat_${step}_$ts.txt"
Select-String -Path "$out\logcat_${step}_$ts.txt" -Pattern 'ATMR_LTC_P|ATMR_JS_J|Unregister|removeLocationUpdates|FLP_|FGS_' |
  ForEach-Object { $_.Line } > "$out\markers_${step}_$ts.txt"
docker exec atmrstg-backend-1 python /tmp/_f03_ages.py > "$out\ages_${step}_$ts.txt"
```

Ou run orchestré : [`_samsung_canary_30min/run_step.ps1`](./_samsung_canary_30min/run_step.ps1)

---

## Transitions mission (entre C02 et C03)

**Option A — dashboard entreprise** (recommandé terrain) :

1. Créer réservation · assigner driver 20 · statut **ASSIGNED** → C03/C04.
2. Chauffeur : **En route** → C05/C06.
3. **Arrivé** → C07/C08.
4. **À bord** / IN_PROGRESS → C09/C10.
5. **Terminée** → C11.

**Option B — staging script** (gain de temps) :

```powershell
# EN_ROUTE direct pour sauter assignation lente
docker cp docs\ops\_samsung_canary_30min\_samsung_canary_mission_enroute.py atmrstg-backend-1:/tmp/
docker exec atmrstg-backend-1 python /tmp/_samsung_canary_mission_enroute.py
```

Puis transitions **ARRIVED / IN_PROGRESS / TERMINAL** depuis l’app chauffeur.

---

## Détail par étape

| ID | App | Mode attendu | P9 cible | Gate critique |
|----|-----|--------------|----------|----------------|
| C01 | FG | `availability_presence` | ~60 s | FGS ON · PG 2+ MATCH |
| C02 | BG | `availability_presence` | ~60 s | HOME Launcher · FGS ON |
| C03 | FG | `mission_live` | ~20 s | `app_resume` interval 20000 |
| C04 | BG | `mission_live` | ~20 s | **pas HMR** · pas wipe scheduling |
| C05 | FG | `mission_live` EN_ROUTE | ~20 s | status ≠ STOP |
| C06 | BG | `mission_live` EN_ROUTE | ~20 s | POST_HOME P8/J1/J7 continuent |
| C07 | FG | `mission_live` ARRIVED | ~20 s | ARRIVED ≠ fin tracking |
| C08 | BG | `mission_live` ARRIVED | ~20 s | idem BG |
| C09 | FG | `mission_live` IN_PROGRESS | ~20 s | — |
| C10 | BG | `mission_live` IN_PROGRESS | ~20 s | ★ gate BG à bord |
| C11 | FG | soft → `availability_presence` | ~60 s | FLP 20→60 · remove=0 · pas restart |

---

## P8 — Carte entreprise (1 min)

URL : `http://localhost:3000/dashboard/company/<company_uuid>`

**PASS** :

- `1/1 en direct` (ou N/N pilote)
- chip **Canary Atmr1**
- `Dernière mise à jour` < 30 s
- coords carte ≈ projection DB

---

## P9 — Verdict final (2 min)

```text
CANARY SAMSUNG 30 MIN — <date>

C01  PASS/FAIL
C02  PASS/FAIL
…
C11  PASS/FAIL
Carte PASS/FAIL

Invariants globaux (tous logcats)
  Unregister A     = 0
  FLP_REMOVE A     = 0
  FGS_restart A    = 0

FIRST_STOP = <étape ou —>
ARTIFACTS  = docs/ops/_samsung_canary_30min/
```

**PASS global** ⇔ C01–C11 + carte + invariants.

Enregistrer dans : [`_samsung_canary_30min/SAMSUNG_CANARY_30_RUN.md`](./_samsung_canary_30min/SAMSUNG_CANARY_30_RUN.md)

---

## Raccourcis monitoring

```powershell
# Ages projection (docker)
docker exec atmrstg-backend-1 python /tmp/_f03_ages.py

# Top activity FG/BG
adb -s $d shell dumpsys activity activities | findstr topResumedActivity

# FGS rapide
adb -s $d shell dumpsys activity services ch.liri.operations | findstr -i foreground
```

Backend (optionnel) :

```bash
bash scripts/staging/capture_canary_metrics.sh staging/output/canary-$(date -u +%Y%m%dT%H%M%SZ).txt
```

---

## Après PASS

- Tag run dans [`gps-production-pilot-closed.md`](./gps-production-pilot-closed.md) (preuve canary récente).
- Continuer FLEET-2 lab (F05+) séparément si besoin.
- **Ne pas** interpréter comme GO PROD général.

## Après FAIL

- Ne pas étendre pilote prod sans RCA.
- Conserver `logcat_*` + `markers_*` + `ages_*` du FIRST_STOP.
- Réessayer **uniquement** après fix ou changement ENV documenté.
