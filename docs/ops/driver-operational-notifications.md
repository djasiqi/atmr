# Notifications opérationnelles chauffeur

Les pushes chauffeur partent des transitions du transport, pas d'un écran web.

| Événement | Destinataire | Texte |
| --- | --- | --- |
| `DRIVER_ASSIGNED` | nouveau chauffeur | Nouveau transport assigné |
| `DRIVER_UNASSIGNED` `reason=reassigned` | ancien chauffeur, A vers B | Ce transport a été réattribué à un autre chauffeur. |
| `DRIVER_UNASSIGNED` `reason=unassigned` | ancien chauffeur, A vers personne | Ce transport ne vous est plus assigné. |
| `ROUTE_CHANGED` | chauffeur assigné | Itinéraire modifié |
| `SCHEDULE_CHANGED` | chauffeur assigné | Horaire modifié |
| `BOOKING_CHANGED` | chauffeur assigné | Transport modifié (itinéraire et horaire dans la même mutation) |
| `BOOKING_CANCELLED` | chauffeur assigné avant annulation | Transport annulé |

Une réattribution A vers B envoie un retrait à A et une assignation à B. Une annulation n'envoie pas en plus un retrait.

L'envoi réutilise `services/events/fanout.py` et `send_push_message`. L'`event_id` est stable pour une même mutation (`services/notifications/driver_operational_events.py`). Le texte système ne contient ni patient, ni médecin, ni note médicale.

Le clic ouvre `lirie://driver/bookings/{id}`, puis l'application recharge la course. Un retrait affiche que le transport n'est plus assigné et revient à la liste.

Pas de déploiement tant que la réattribution et l'annulation ne sont pas vérifiées sur iPhone.
