# P0-D D5 — A/B stationnaire Prod126 vs Dev Client

```text
NOM                      = RELEASE-ONLY EXPO LOCATION DELIVERY FAILURE
PRESENCE MODEL           = CLOSED / CORRECT ✅
"stale = immobile"       = RULED OUT ✅
D5-B FLP→Expo delivery   = CONFIRMED DIFFERENTIAL ✅
stationnaire comme cause = RULED OUT ✅
qualité GPS locale       = fortement exclue
MOCK MOVEMENT workaround = RULED OUT ✅
backend/HTTP             = hors cause du cut D5
fake connected UI        = NO-GO
A/B STATIONARY           = FREEZE ✅
PATCH                    = NO-GO
GENERAL DISTRIBUTION     = NO-GO
BACKEND                  = GELÉ (prod) ; staging local OK pour bras DEV
```

RCA suite (4 familles) : [`../d5_release_only/D5_RELEASE_ONLY_DIFF.md`](../d5_release_only/D5_RELEASE_ONLY_DIFF.md)

## Objectif

Même téléphone, même appartement, **sans mock**, ~4 min par bras.

## Grille A/B (FREEZE)

| Signal | PROD126 | DEV125 |
|--------|---------|--------|
| Binary | vc **126** non-DEBUGGABLE Play | vc **125** DEBUGGABLE Dev Client |
| API / driver | prod / **20135** | staging `:15100` / **20** mission **#28** |
| WorkSource uid | **10905** | **10906** |
| Fused request Expo | `@+8s0ms HIGH_ACCURACY`, `minUpdateInterval=0` | **identique** |
| FLP (`GmsPassiveListener_FLP`) | **26** | **26** |
| too close | **234** | **152** |
| too fast | **89** | **65** |
| Location unavailable | **108** | **17** |
| `TaskService: Finished … background-location-task` | **0** | **9** (~toutes ~20 s) |
| hits `background-location-task` (JS+Finished) | **0** | **27** |
| PUT | **0** | **25** (gateway staging 200) |
| LOC PG (fenêtre) | **0** | **≥12** (mission 28) |

Fenêtres :

| Bras | T0 → Tend (+02) | Artefacts |
|------|-----------------|-----------|
| PROD126 | `19:29:18 → 19:33:41` | [`PROD126_BASELINE_FREEZE.md`](PROD126_BASELINE_FREEZE.md) + `prod_*` |
| DEV125 | `20:04:11 → 20:08:32` | [`DEV125_ARM_FREEZE.md`](DEV125_ARM_FREEZE.md) + `dev_*` |

## Verdict

```text
PROD Finished/PUT/LOC = 0
DEV  Finished/PUT/LOC > 0   (9 / 25 / ≥12)

→ D5 = RELEASE-ONLY EXPO LOCATION DELIVERY FAILURE
→ stationnaire comme cause exclusive = RULED OUT
→ heartbeat « faute de fix » non justifié par cet A/B seul
  (Dev livre en stationnaire ; Prod ne livre pas)
```

Request dumpsys **identique** → discriminant **pas** la forme `ProviderRequest`.  
Diff structurel suivant (voir RCA) : **LocationTaskService** Prod `startForegroundCount`/`ConnectionRecord` **×100** vs Dev **×1**.

## Caveats

- Dev = DEBUGGABLE + Metro + staging (pas Play vs Play).
- Driver / mission / backend différents.
- Compteur timeline `finishedish=27` = toutes mentions ; **Finished TaskService strict = 9**.

## Protocole

1. ✅ `PROD` — 240 s — FREEZE
2. ✅ Uninstall Prod126 / install Dev 125
3. ✅ re-login staging + mission #28 ([`MISSION_ATMR1.md`](MISSION_ATMR1.md))
4. ✅ `DEV` — 240 s — FREEZE
5. ⏳ Réinstaller Prod 126 si besoin ops — **pas de patch**
