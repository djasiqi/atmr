# P0-E — Canary 135 #2 post `pm clear` — verdict (2026-08-17)

## Statut

```text
pm clear                     = Success ✅
versionCode                  = 135 ✅
re-login 20135 / mission     = OK ✅
session neuve                = trk_sess_1786985556979_ypmkdr5z (gen 1706) ✅
pré-canary                   = PASS ✅
  DLE au gate                = 7 (seq 1→7)
  OTHER_SESS post-clear      = [] ✅
  canon match session        = OK

CANARY 135 #2 HOME 120s      = FAIL gate ❌
FIX immutabilité 135         = NON INVALIDÉ ★
event_id_payload_conflict    = 0 ✅ (fenêtre HOME + 3m consumer)
DLE Δ HOME                   = 0 ❌ (figé seq=7 / id=6449)
canonical Δ                  = 0 ❌
REST                         = stale → offline ❌
FGS / Finished / PUT 202     = continuent ✅
SERVER                       = NE PAS TOUCHER ✅
PLAY                         = HOLD ⛔
Q1                           = HOLD
```

## Discriminant demandé (post-clear)

Tous les DLE de la session active sont **créés après** le `pm clear` (started_at `16:52:36Z`).  
Aucun conflit consumer sur cette fenêtre → le poison pré-135 **n’explique plus** un `event_id_payload_conflict`.

```text
conflict post-clear          = 0  → fix immutabilité non contredit
DLE/canonical morts          = autre cause (voir ci-dessous)
```

## Timeline

| Phase | UTC | Observé |
|-------|-----|---------|
| Session start | 16:52:36Z | active gen 1706 |
| Promote seq 1→7 | 16:52:37 → 16:53:46Z | `[p5b_promote]` OK |
| Dernier DLE | 16:53:46Z | seq=7 eid=`…6vjrmxcq` |
| FG warm canary | 16:54:36Z+ | PUT 202, FGS OK, **déjà plus de promote** |
| HOME | 16:55:06 → 16:57:06Z | PUT ~3/20s, Finished OK, conflict=0, DLE figé |
| Device | pendant / après HOME | `Location unavailable for foreground-service task delivery` |

## Lecture

1. **Immutabilité (cible fix 135)** — sur ledger neuf, `event_id_payload_conflict = 0` alors que les PUT retry continuent → le scénario « même eid + payload muté → DLQ » **n’apparaît plus**. Le fix n’est **pas invalidé** ; il est **soutenu** par ce run (sous réserve d’un re-canary avec captures GPS fraîches).

2. **BG_FRESHNESS end-to-end** — **non validé** : DLE/canonical/REST ne progressent pas. Le gel commence **avant** HOME (~1 min après login), pas seulement en background.

3. **Cause candidate dominante (run #2)** — plus de `[p5b_promote]` après seq 7 + `Location unavailable` côté TaskManager + PUT 202 stables (taille 346, lots de 3) → hypothèse : **peu/pas de nouvelles captures GPS** ; la file rejoue des items déjà vus (payload immuable → pas de conflict, pas de nouveau DLE). Pont Q1 (`ingested_non_persisted` / awaiting durable) reste plausible en parallèle, mais **hors invalidation du fix 135**.

## Artefacts

- `docs/ops/_p0e_bg_freshness_135_2_2026-08-17/` (timeline, server_timeline, put_ack, consumer, logcat)
- `docs/ops/_p0e_precinary_135_2.py`
- `docs/ops/gps-p0e-canary-135-2-pmclear-2026-08-17.md`

## Next (sans toucher serveur)

```text
1. Assurer GPS réel (extérieur / fix Location unavailable)
2. Confirmer en FG que DLE/canonical avancent (seq > 7)
3. Re-canary HOME 120 s #3
4. Si conflict=0 + DLE/canon OK → BG_FRESHNESS FIX VALIDATED → reprise Q1
5. Si conflict>0 sur eid post-clear → fix immutabilité incomplet (FIRST eid PG vs retry)
```

## ✅ Implémenté

- `pm clear` + verify 135 + re-login + session neuve
- Gate pré-canary PASS
- Canary HOME 120 s exécuté
- Verdict #2 documenté (conflict=0 ; gate fraîcheur FAIL ; fix non invalidé)
