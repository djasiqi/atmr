# P0-E Phase 2 canary — verdict 2026-08-17 (GO après gate PASS)

```text
ACTIVE SESSION GATE (pré) = PASS ✅ (lauam301)
RE-GO PG_FIRST            = DONE (true ~13:23–13:29Z)
ATTRIBUTION P5-B          = FAIL ⛔
ROLLBACK PG_FIRST         = false (en cours / fait)
GLOBAL ENABLE             = NO-GO ⛔
RC132                     = FROZEN ✅
```

## Ce qui a marché

- Soft préconditions / enable : `PG_FIRST=true`, OUTBOX=true, healthy, TB=0
- Session pré-canary `trk_sess_1786972692514_lauam301` était active avec DLE 6060–6067

## Pourquoi l’attribution a échoué

Pendant la fenêtre canary :

1. `lauam301` est passée **superseded** (nouvelles sessions actives `…3zzbvuqa` / `…gdnf3xtm`)
2. Les DLE ont **continué** sur `lauam301` (superseded) — ex. 6069–6073
3. Session **active** `trk_sess_1786973176090_gdnf3xtm` : **0 DLE**
4. Annexe A.3 : `superseded` → `publish_realtime=false` → **pas de `_maybe_promote_after_pg`**
5. Redis `loc:canonical` présent **sans** gen/seq = writer sync LocationService ≠ P5-B

```text
PG avance (souvent)     ✅
mais sur session superseded
→ promote skip          ✅ (by design)
→ preuve P5-B           ⛔
```

## STOP / rollback

Pas de traceback / unhealthy. Échec = **aucune LOC sur session active** pendant la fenêtre → canary non concluant → **PG_FIRST=false**.

## NEXT

Le bloquant reste mobile : flush ledger sur sessions superseded alors qu’une session active existe sans points.

Options :
1. **HOLD P5-B** — prochain chantier : SESSION OWNERSHIP / ROTATION (Q3-A/Q3-B)
   → `docs/ops/gps-p0e-session-ownership-q3-2026-08-17.md`
   (pas de 3e canary PG-first tant que rotate mid-fenêtre + drain superseded)
2. Puis re-GO fenêtre très courte dès la 1re DLE active
3. Ne pas enable global

Idempotence serveur / RC132 : inchangés.
