# P0-E — GO build interne 134 (Q1 RCA observabilité ONLY)

## Statut figé

```text
Q1 STATIC ROOT               = ATTRIBUTED ✅
Q1 RUNTIME ROOT              = HOLD ⛔ (pause — BG freshness prioritaire)

BUILD 134 QA                 = EAS IN_PROGRESS (install après GO reprise)
ACK BEHAVIOR PATCH           = HOLD ⛔
event_id conflict            = ROOT BG_FRESHNESS ★★★ (+ associé Q1)
PLAY                         = HOLD ⛔

BG_FRESHNESS (HOME→0/8)      = ATTRIBUTED ✅
  docs/ops/gps-p0e-bg-freshness-rca-2026-08-17.md
  ROOT = event_id_payload_conflict → 0 DLE → canonical figé → REST stale

L12 lifecycle (133)          = PASS ✅ (Android OK ; map stale ≠ arrêt GPS)
```

## Contenu du build (uniquement obs)

Panneau QA missions — section **Q1 ACK (bridge)** :

```text
lastAckStatus
lastAckError
lastAckSeq          (= lastAckAttemptSeq bridge)
lastAckEventId
currentSeq          (= currentAttemptSeq)
currentEventId      (= currentAttemptEventId)
queueDepth
```

Pas de changement de sémantique ACK / `resolveBridgeAckFields` / labels produit.

## Commande

```bash
cd mobile/unified-app
eas build --platform android --profile production-apk --non-interactive
```

## Capture décisive (après install)

Sur **un seul eid X** :

```text
T0  enqueue X
T1  PUT X → 202
T2  client → ingested_non_persisted

T3  QA PANEL immédiatement :
    lastAckStatus = ?
    lastAckError  = ?
    lastAckEventId = X

T5  serveur : PG contient exactement X ✅

T6  watermark / persistence backend confirmée

T7  QA PANEL après persistence :
    lastAckStatus = ?
    lastAckError  = ?
    lastAckEventId = ?
```

### Verdict smoking gun → RCA CLOSED

```text
T3 lastAckError = ack_ingested_non_persisted
T5 X persisté PG ✅
T7 lastAckError reste ack_ingested_non_persisted
   (ou aucun ACK final ne remplace)

→ Q1 ROOT = ACK SEMANTIC MISMATCH CLIENT ✅
→ RCA Q1 = CLOSED ✅
```

Sinon (T7 = persisted/accepted mais UI incorrecte) → descendre dans `formatBridgeSyncLabel` / corrélation seq-eid.

**Source de vérité** = QA `lastAck*` (pas le texte header « Envoyé » / « Non confirmé »).

## Smoke install

```text
adb shell dumpsys package ch.liri.operations | grep versionCode
→ 134
```

Ouvrir Missions → panneau Tracking QA → section Q1 ACK visible.

## ✅ Implémenté

- `app.json` versionCode **134**
- `DriverTrackingQaPanel` : champs Q1 ACK exposés (noms capture)
- EAS `production-apk` déjà `EXPO_PUBLIC_TRACKING_QA_PANEL=1`
- ACK behavior : **non modifié**
