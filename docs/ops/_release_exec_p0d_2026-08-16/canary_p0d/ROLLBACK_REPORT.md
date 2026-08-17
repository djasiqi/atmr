# Rollback hot-patch canary P0-D — 2026-08-16

```text
PROD officiel              = 286737a2 (image inchangée)
HOT-PATCH LIVE             = ROLLBACK ✅
D4-B patch                 = VALIDÉ PARTIELLEMENT (conservé en code repo)
D4-B conclusions           = INCHANGÉES ✅
ANDROID 126 HOME/BG        = FAIL ❌
GENERAL DISTRIBUTION       = NO-GO ❌
```

## Exécution

| Item | Valeur |
|------|--------|
| Backup source | `/tmp/atmr-p0d-canary-20260816T151955Z-backup` |
| Script | `rollback_canary_p0d.sh` |
| Sortie | `rollback_out.txt` → `ROLLBACK_OK` |
| Image | `djasiqi/atmr-backend:sha-286737a2362e` |

Restauré :

- `persist_with_outbox.py` (backend + consumer) — sans `compare_persisted_event`
- `driver.py` (backend) — défaut `recorded_at` pré-P0-D (`ts` ou `now`)
- suppression de `location_idempotency.py` dans les deux containers

Vérifs :

```text
HAS_COMPARE (grep)     = 0
IDEM_GONE              = yes
CONSUMER_IDEM_GONE     = yes
recorded_at manquant…  = présent (driver legacy)
```

## Artefacts canary conservés (ne pas purger)

Sous `docs/ops/_release_exec_p0d_2026-08-16/canary_p0d/` :

- `CANARY_P0D_REPORT.md`
- `smoke_timeline.txt`, `snap_*.txt`, `dlq_*.txt`, `raw_post.txt`
- `deploy_canary_out.txt`, `analyze_out.txt`, `put_correlate.txt`
- copies `prod_*` / scripts deploy+rollback

## Conclusions figées (post-rollback)

```text
D4-B BUG CONFIRMÉ ✅
  recorded_at mutable → event_id_payload_conflict

D4-B CORRECTION CONFIRMÉE ✅
  recorded_at stable → conflits supprimés (FG)

HOME/LOCK FAIL ❌
  panne AVANT HTTP (0 PUT après 15:23:19Z)
  → hors D4-B serveur
```

## Suite

```text
BACKEND                = ne plus toucher (prod = tip image 286737a2)
P0-D suite             = Location/TaskManager → task → JS → enqueue → HTTP
SMOKING GUN suivant    = task Expo encore invoquée après cut ?
GENERAL DISTRIBUTION   = NO-GO
```
