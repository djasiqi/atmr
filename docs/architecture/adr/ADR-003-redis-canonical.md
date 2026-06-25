# ADR-003 — Redis canonical source temps réel

## Statut

Accepté — 2026-06-25

## Contexte

Risque d'écritures parallèles (stub internal_tracking, double paths).

## Décision

**Redis** `driver:{id}:loc:canonical` est la seule source temps réel serveur. Postgres = historique. Pipeline unique : ingest → persist → fanout.

## Conséquences

- Stub `internal_tracking.py` branché sur `enqueue_tracking_event`
- INV-5 vérifié par architecture linter
- Frontend lit Socket/HTTP API uniquement (INV-4)
