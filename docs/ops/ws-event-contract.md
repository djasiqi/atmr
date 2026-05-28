# Contrat événements WebSocket (backend → ws-service)

| event_type | room_target | transport | criticality | dedup_required |
|------------|-------------|-----------|-------------|----------------|
| booking_updated | company_{id} | redis_relay | critical | yes |
| booking_cancelled | company_{id} | redis_relay | critical | yes |
| team_chat_message | company_{id} | redis_relay | critical | yes |
| team_chat_typing | company_{id} | redis_relay | normal | no |
| dispatch_assignment | company_{id} | redis_relay | critical | yes |
| dispatch_run_started | company_{id} | redis_relay | critical | yes |
| dispatch_run_completed | company_{id} | redis_relay | critical | yes |
| dispatch_run_failed | company_{id} | redis_relay | critical | yes |
| driver_location_update | company_{id} | kafka | high | yes |
| driver_live_state_update | company_{id} | kafka | high | yes |

Payload requis : `event_id`, `timestamp` (via `SocketEvent.create`).
