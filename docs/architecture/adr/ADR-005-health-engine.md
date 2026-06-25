# ADR-005 — TrackingHealthEngine

## Statut

Accepté — 2026-06-25

## Contexte

Dashboard et ops manquaient d'un état agrégé corrélé mobile ↔ backend.

## Décision

**TrackingHealthEngine** calcule `HEALTHY | WARNING | DEGRADED | BROKEN` toutes les 30 s (Celery). API `/api/v1/companies/me/tracking-health`. Frontend merge `tracking_health_state`.

## Conséquences

- Alignement badge « Non localisé » sur Health Engine
- Alertes Prometheus corrélées
- Pas de logique parallèle stale côté frontend
