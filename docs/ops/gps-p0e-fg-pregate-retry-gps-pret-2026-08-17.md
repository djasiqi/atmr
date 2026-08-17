# P0-E — FG PRE-GATE retry (« GPS prêt ») — FAIL → HOME #3 non lancé

## Verdict

```text
SIGNAL                       = GPS prêt
FG PRE-GATE retry            = FAIL ❌
HOME #3                      = NON LANCÉ ⛔
conflict                     = 0 ✅
IMMUTABILITY 135             = soutenu ✅
PATCH UX / SERVER / PLAY     = HOLD / inchangé / HOLD
```

## Observations (captures réelles uniquement)

### Tentative A — session attendue `…ypmkdr5z`

```text
FAIL session_mismatch
got    = trk_sess_1786989187108_ojihpgbv (gen 1708, started 17:53:07Z)
expect = trk_sess_1786985556979_ypmkdr5z
```

La session témoin a changé (hors protocole « ne pas créer de nouvelle session ») — noté, pas de purge serveur.

### Tentative B — session active actuelle `…ojihpgbv` (90 s FG)

```text
DLE n / max_seq              = 0
new_event_id                 = 0
recorded_at                  = None (aucune DLE)
canonical                    = absent (ttl=-2)
REST                         = last_known (age ~3500 s)
seq_delta                    = 0
```

**Aucune nouvelle capture réelle** pendant 90 s app au premier plan.

## Lecture P-TECH

```text
P-TECH = CONFIRMÉ côté FG ★
→ pas d’avancement event_id / recorded_at
→ le problème est avant / hors enqueue utile de nouvelles positions
→ pas Redis, pas immutabilité 135, pas encore un fail HOME

PUT 202 / retries            = IGNORÉS comme critère (cadre)
HOME #3                      = HOLD (pré-gate rouge)
```

Preuve plus forte qu’après #2 : même en FG, session active, **0 DLE** sur la session courante.

Corrélation pendant la fenêtre :

```text
PUT /location (~2 m)         = 45  (trafic HTTP — IGNORÉ comme PASS)
p5b_promote (~10 m)          = 0 lignes
Location unavailable         = présent (logcat)
```

Confirme : **trafic ≠ nouvelles captures** ; delivery/production de fixes Location en FG est le discriminant.

## Artefacts

- `docs/ops/_p0e_bg_freshness_135_3_2026-08-17/fg_pregate_retry2.txt` (mismatch)
- `docs/ops/_p0e_bg_freshness_135_3_2026-08-17/fg_pregate_retry2b.txt` (FAIL 0 eid)

## Next

```text
→ ouvrir P-TECH delivery/production Location (FG) — diagnostique mobile/Expo/FLP
→ ne pas lancer HOME #3
→ ne pas patcher UX (P-UX) dans ce train
→ conflict=0 reste = fix 135 non remis en cause
```
