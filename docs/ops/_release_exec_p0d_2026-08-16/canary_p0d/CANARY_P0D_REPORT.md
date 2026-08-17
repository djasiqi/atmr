# Canary P0-D serveur + smoke build 126 — 2026-08-16

```text
MOBILE 126                 = INCHANGÉ ✅ (1.0.11 / versionCode 126)
BACKEND P0-D               = HOT-PATCH CANARY ✅
MIGRATION / PURGE          = AUCUNE ✅
GENERAL DISTRIBUTION       = NO-GO ❌
```

## Déploiement canary

| Item | Valeur |
|------|--------|
| Stamp | `p0d-canary-20260816T151955Z` |
| Backup | `/tmp/atmr-p0d-canary-20260816T151955Z-backup` |
| Cibles | `atmr-backend-1` + `atmr-tracking-kafka-consumer-1` |
| Fichiers | `location_idempotency.py`, `persist_with_outbox.py` (base **prod** sans colonne `capture_id`), `driver.py` (API) |
| Health post-restart | backend healthy, consumer healthy |
| Verify | `HAS_COMPARE=True`, `HAS_CAPTURE_COL=False` |

Méthode : hot-patch `docker cp` + restart (pas de rebuild image / pas de SHA tip).  
Image sous-jacente reste `djasiqi/atmr-backend:sha-286737a2362e`.

## Smoke (USB `RFCW20QC53W`, driver 20135, mission 38224)

| Phase | Durée | Résultat |
|-------|-------|----------|
| FG | 120 s | LOC OK — 47 rows post-deploy jusqu’à seq **71** |
| HOME | 120 s | **LOC = 0** ; **PUT = 0** après 15:23:19Z |
| LOCK | 60 s + reprise | **LOC = 0** |

P0-A/B signaux logcat : `auth_not_usable=0`, `native_start_error=0`, `overlap=0`, `gen_null=0`.

## Preuve causale D4-B (FG / retries)

Sur `raw.v2` après patch, retries du même item :

```text
eid=trk_1786889838323_9mbhiy9q  seq=71
recorded_at = 2026-08-16T14:17:17.016Z   (STABLE)
sent_at     = 15:23:16 → 15:23:19        (variable)
```

→ l’ingress `timestamp` → `recorded_at` **tient**.  
Derniers conflicts DLQ `event_id_payload_conflict` : **15:21:24Z** (backlog Kafka **pré-patch** avec `recorded_at` déjà mutés) — plus aucun conflict ensuite pendant le flush FG jusqu’à 15:23:19.

```text
MULTI_ROW_SAME_EID depuis deploy = []
PUT 202 pendant flush FG         = oui (pic ~15:22–15:23)
```

## Critères décisifs

| Critère | Résultat |
|---------|----------|
| `event_id_payload_conflict` sur retries **post-ingress** (recorded_at stable) | **0** observé après drain backlog |
| DLQ D4-B pendant HOME | N/A — **plus de PUT** |
| LOC PG pendant HOME/BG/lock | **NON (0)** |
| HOL / orphan / P0-A / P0-B / gen=null | **0** |

## Verdict

```text
D4-B SERVER PATCH (ingress + idempotence)
= PARTIAL PASS ✅ sur FG
  → retries à recorded_at stable + sent_at variable
  → persistence jusqu’à seq 71
  → plus de conflict une fois le poison pré-patch drainé

SMOKE HOME/BG/lock build 126
= FAIL ❌
  → arrêt net des PUT ~15:23:19Z (avant/au début HOME)
  → pas de chaîne device→HTTP pendant HOME
  → D4-B n’est PAS l’unique cause du smoke FAIL

FGS / delivery secondaire
= REOPEN as LEADING for HOME cut
  (DENIED / task delivery — hors périmètre patch serveur seul)

ANDROID PRODUCTION BINARY
= NOT READY ❌

GENERAL DISTRIBUTION
= NO-GO ❌
```

## Lecture RCA

Le canary **prouve** que D4-B était un bug serveur réel et que le patch corrige les retries HTTP idempotents quand le client envoie encore.

Il **infirme** l’hypothèse « D4-B seul explique l’absence de LOC HOME » : dès que les PUT cessent, aucune correction consumer ne peut produire de LOC.

## Artefacts

- `deploy_canary_out.txt`, `smoke_timeline.txt`
- `snap_*.txt`, `dlq_*.txt`, `raw_post.txt`, `analyze_out.txt`, `put_correlate.txt`
- `run_canary_smoke_p0d.ps1`

## Suite recommandée

1. ~~Rollback hot-patch~~ → **FAIT** (`ROLLBACK_REPORT.md`)
2. Rouvrir diagnostic **FGS / TaskService delivery HOME** (build 126) sans rouvrir A/B/ledger ; **backend gelé**.
3. Ne pas passer READY FOR DISTRIBUTION tant que HOME n’a pas de PUT→LOC.
4. Réintégrer D4-B via release immuable (pas de hot-patch) quand le cut HOME sera classé.
