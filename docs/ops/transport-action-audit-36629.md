# Audit existant — cas #36629 / #3224 et chemins de mutation

**Date** : 2026-07-21  
**Référence** : [`transport-decision-workflow.md`](../domain/transport-decision-workflow.md)

## Constat DB local (#36629 / #3224)

Environnement Docker local interrogé :

```sql
SELECT ... FROM booking WHERE id = 36629;  -- 0 rows
SELECT ... FROM transport_requests WHERE id = 3224;  -- 0 rows
```

Les identifiants du cas prod **ne sont pas présents** dans la base de développement locale. Aucune réparation automatique n’a été effectuée.  
Sur prod, vérifier manuellement : `booking.status`, `driver_id`, `cancellation_*`, `active_change_request_id`, `transport_request.status`, timeline `cancelled`, assignments.

Symptômes rapportés (UI) :
- Institution : timeline « Transport annulé » + badge encore « Chauffeur assigné »
- Entreprise : historique « Modification institution » + statut « Assignée »

Cause code la plus probable : annulation institution immédiate **sans** fanout / sans sync request / libellé company générique ; ou désync statut vs timeline.

## Inventaire — mutations institution post-engagement

| Path | Classification | Action V1.1 |
|---|---|---|
| `cancel_institution_booking` | MUST_DEPRECATE (immédiat) | → intention `CANCELLATION` |
| `update_institution_booking` patch direct (flag off / mineurs / legs) | MUST_DEPRECATE si contractuel | → intention `CHANGE_*` (strict) |
| `create_change_request` (PR2) | KEEP → évolue en TransportAction | Enrichir + exchanges |
| `RespondToChangeRequestUseCase._refuse` (apply+redispatch) | MUST_DEPRECATE sémantique | → REJECT no-op mission |
| `RespondToChangeRequestUseCase._accept` | KEEP → EffectPlan | Appliquer via workflow |
| Release/redispatch endpoint institution | KEEP (action explicite séparée) | Hors décision cancel/modif |
| Cancel demande pré-CONVERTED | KEEP | Direct OK |
| Company / driver cancel & assign | KEEP | Hors scope institution |

## Chemins à protéger par tests anti-régression

1. Post-engagement : `POST .../bookings/{id}/cancel` ne passe plus le booking en `CANCELED` immédiatement.
2. Post-engagement : patch institution ne mute plus le booking avant ACCEPT entreprise.
3. Refuse entreprise : `booking.status` / adresses / horaire inchangés.
4. Accept cancel : atomicité booking + request + driver + assignments + outbox Completed.
