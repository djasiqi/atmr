# ADR-001 — Tracking Engine unique (mobile)

## Statut

Accepté — 2026-06-25

## Contexte

Double capture (watch + FGS + legacy HTTP) sans orchestrateur a produit des états zombie et des ReferenceError.

## Décision

Un **Tracking Engine** unique (`driverTrackingBridge` + engines dérivés) orchestre capture, queue et envoi. Les booléens isolés deviennent dérivés de la FSM.

## Conséquences

- `PresenceTrackingEngine` / `MissionTrackingEngine` pour séparation présence vs mission
- `sendLegacyPoint` réservé au mode sans queue persistent
- Tests FSM obligatoires en CI
