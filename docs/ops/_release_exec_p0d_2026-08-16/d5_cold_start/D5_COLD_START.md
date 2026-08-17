# P0-D D5 — Cold start (figé / partiellement abandonné)

```text
FORCE-STOP TEST      = ABANDONNÉ pour D5 ✅
PRE_FORCE 1/1        = PREUVE VALIDE ✅
COLD-START suite     = remplacé par session normale
→ ../d5_session_normal/D5_SESSION_NORMAL.md
```

## Preuve conservée

```text
PRE_FORCE (avant tout force-stop) :
  startForegroundCount = 1
  binds                = 1
  isForeground         = true
  startRequested       = true

→ Prod126 ne démarre PAS naturellement avec le storm ×100
```

Les runs `force-stop → FGS absent` restent des **artefacts de protocole**, pas des datapoints D5.

## Suite

Observer en **usage normal** le premier `1→2` (sans force-stop).
