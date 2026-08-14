# P0-C-LEDGER — Diagnostic read-only #3 (bodies exacts des 3 queue_item_id)

```text
GO                         = READ-ONLY body HTTP / queue SQLite
PATCH                      = NO-GO
CLAIM REDIS                = non touché
SOURCE                     = device SQLite files/SQLite/driver_tracking_queue_v5.db
EXPORT                     = docs/ops/_c3_ab_2026-08-14/p0c_ledger_three_items.json
MAPPING HTTP               = mobile/.../features/driver/api.ts sendDriverLocation
```

## Freeze

```text
Redis claim leak cycle          CONFIRMED ✅
duplicate != persisted          CONFIRMED ✅
ledger_ids_missing              CONFIRMED ✅

CAUSE DES IDs MANQUANTS
L1 (generation null)            CONFIRMED ✅
L2 sérialisation                EXCLUDED
L3 parsing backend              EXCLUDED
L4 sequencing causal            CONTRIBUTING (pas causal pour ces 3)
L5 body valide / ACK erroné     EXCLUDED

PATCH                           NO-GO
```

---

## Reconstruction des 3 items (queue = vérité terrain)

Tous partagent :

```text
tracking_session_id   = trk_sess_1786722711491_2h9hb2ps
session_generation    = NULL          ← discriminant
app_state             = background
location_mode         = mission_live
mission_id            = 26
state                 = ingested_non_persisted
delivery_state        = retry_pending
```

| queue_item_id | created (local) | sequence | GNSS timestamp (UTC) | lat / lng | accuracy |
|---------------|-----------------|----------|----------------------|-----------|----------|
| `trk_1786723792342_u8w2gqur` | 18:09:52 | **53** | 16:09:50.647Z | 46.2116082 / 6.1262536 | 15.3 m |
| `trk_1786723810647_11n415gl` | 18:10:10 | **54** | 16:10:09.291Z | 46.2115969 / 6.1262051 | 14.9 m |
| `trk_1786723829101_tdhsi20c` | 18:10:29 | **55** | 16:10:27.839Z | 46.2116322 / 6.1262227 | 12.7 m |

Coords **≠** ancre 17:20:10 (46.19015 / 6.14455) → **nouveaux fixes GNSS réels** à ~18:09–18:10, bloqués ensuite par C-LEDGER (pas un replay d’ancre).

Session watermark polluée : même `trk_sess_1786722711491_2h9hb2ps` → HTTP **403** en reprise (diag #1).

---

## Body HTTP reconstruit (fidèle au code client)

`sendDriverLocation` envoie notamment :

```text
tracking_session_id  ← item.trackingSessionId   (présent)
session_generation   ← item.sessionGeneration   (null)
sequence_id          ← item.sequenceId          (53/54/55)
tracking_event_id    ← item.id
+ lat/lng/accuracy/timestamp/mission_id/...
Idempotency-Key / X-Location-Event-Id = event id
```

Exemple (item 1) :

```json
{
  "latitude": 46.2116082,
  "longitude": 6.1262536,
  "accuracy": 15.289999961853027,
  "mission_id": 26,
  "timestamp": "2026-08-14T16:09:50.647Z",
  "location_mode": "mission_live",
  "is_background": true,
  "tracking_event_id": "trk_1786723792342_u8w2gqur",
  "tracking_session_id": "trk_sess_1786722711491_2h9hb2ps",
  "session_generation": null,
  "sequence_id": 53,
  "capture_id": "cap_mst57phw_gzvue66s7w"
}
```

Les trois bodies sont **structurellement identiques** : même session, `generation=null`, sequences contigues 53→55.

---

## Create vs retry

```text
queue persistée (création)     generation=null, seq=53/54/55, session fixe
→ dequeue / HTTP               mêmes champs (mapping 1:1)
→ retry/requeue                pas de perte d’IDs observée
```

Les IDs **ne disparaissent pas** entre queue et retry : `session_generation` était **déjà null à l’enqueue**.

Cause côté client (code, read-only) :

```text
createLocalTrackingSession()
  → sessionGeneration = null
  → sequenceCounter = 0
registerSessionWithBackend()  (async)
  → seulement alors sessionGeneration = res.session_generation
enqueue() capture this.sessionGeneration  (peut rester null)
  → points existants jamais réécrits (« Ne jamais réécrire … »)
```

---

## Classification L1–L5

| Cas | Verdict | Preuve |
|-----|---------|--------|
| **L1** queue sans identité ledger | **CONFIRMED** (generation) | `session_generation=NULL` dans SQLite + HTTP |
| **L2** IDs queue OK, absents HTTP | **EXCLUDED** | mapping fidèle ; null reste null |
| **L3** HTTP OK, backend lit absent | **EXCLUDED** | `extract_sync_ledger_ids` : `session_generation is None` → `ledger_ids_missing` correct |
| **L4** seq=1 / session non réconciliée | **CONTRIBUTING** | pour **ces 3** : seq=53–55 (pas L4 direct) ; le churn seq=1+gen null sur items récents reste un facteur systémique |
| **L5** body valide, ACK erroné | **EXCLUDED** | body incomplet → ACK justifié |

### Lien C-SEQUENCING ↔ C-LEDGER

Pas « seq=1 sur ces trois » comme cause.  
Lien réel : **session locale créée sans `session_generation` backend** (register incomplet / 403) + enqueue pendant `generation=null` → items ledger-incomplets durables → cycle claim/ACK.

---

## Contexte client (synthèse)

```text
tracking session courante (items) = trk_sess_1786722711491_2h9hb2ps
native / FGS                      = vivant à ~18:10 (nfix=0 health)
mission                           = 26 EN_ROUTE, mode mission_live
queue insertion                   = background (app_state=background)
nouvelle session                  = session déjà churnée plus tôt (17:51 ms ID) ;
                                    generation jamais hydratée avant enqueue 53–55
```

---

## Implémentation

✅ **Implémenté** : extraction SQLite device read-only ; reconstruction HTTP ; comparaison des 3 bodies ; L1–L5 classés ; export JSON.  
**Reste à faire** : pourquoi `registerSessionWithBackend` n’a pas hydraté cette session (403 watermark) — lecture suivante optionnelle ; **PATCH NO-GO**.
