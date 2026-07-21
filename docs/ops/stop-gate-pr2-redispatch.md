# Stop-gate PR2 — OBSOLÈTE

> **Statut :** obsolète depuis 2026-07-21.  
> Remplacé par [`docs/domain/transport-decision-workflow.md`](../domain/transport-decision-workflow.md).

## Ce qui change

| PR2 (ancien) | TransportAction (nouveau) |
|---|---|
| Refuse = appliquer patch + redispatch | Reject = mission inchangée |
| Cancel institution immédiat | Intention CANCELLATION → décision entreprise |
| Révalidation MAJOR_FIELDS seulement | Modèle strict post-engagement |
| Events ad hoc | `TransportActionCompleted` + LegacyAdapter |

Ne plus se fier à ce document pour l’implémentation.
