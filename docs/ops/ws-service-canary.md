# Runbook — ws-service canary (Phase 2 PR C)

## Variables

| Variable | Staging | Prod pré-canary | Prod canary |
|----------|---------|-----------------|-------------|
| WS_RELAY_PUBLISH_ENABLED | true | false | **true** |
| WS_SERVICE_ACCEPT_CONNECTIONS | true | true | true / kill switch |
| EXPO_PUBLIC_WS_CANARY_ENABLED | 1 | 0 | rollout % |

## Kill switch

1. `WS_SERVICE_ACCEPT_CONNECTIONS=false` ou `POST /ops/ws/kill-switch` sur ws-service
2. Drain 30 s puis force disconnect des SID restants
3. Flag mobile `ws_service_canary` → 0 %
4. Rollback compose si nécessaire (fragment PR D)
5. Reset après diagnostic : `POST /ops/ws/kill-switch/reset` (in-memory uniquement)

### Validation locale (kill switch testé via harness Phase 2)

- `accept_connections=false` reflète à la fois la variable d'env ET le flag in-memory
- `force_disconnect_total` exposé dans `/health` pour audit
- Drain force-disconnect testé : DRAIN_SEC=5 → clients déconnectés à t+5s

## Mode dégradé Kafka

- `POST /ops/kafka/degraded?engaged=true` engage manuellement (ops/tests)
- `/health.kafka_degraded` expose l'état
- Trigger automatique via `WS_KAFKA_LAG_CRITICAL_THRESHOLD` : **dette P1** — non implémenté.
  La pause `WS_KAFKA_DEGRADED_PAUSE_SEC` existe mais n'est déclenchée que manuellement.
- La websocket loop n'est jamais bloquée par Kafka (asyncio.Task séparée — vérifié).

## Rollback canary auto

- Dogfood / 5 % : `confirmed_critical_miss > 0` ou circuit relay open
- 20–50 % : `miss_rate > 0.05%` sur 5 min

## Gate polling

Tester iOS/Android avec `websocket` + `polling` et header `X-WS-Canary`.

## Resync REST (bloquant PR D)

Après 30 s sans event WS : `company_data_stale_resync` + `GET /api/v1/companies/me/drivers/locations`.
