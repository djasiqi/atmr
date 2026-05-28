# Phase post-PR D — ws-service multi-instance

Non bloquant pour PR D (1 replica initial).

## Questions design

- Sticky Traefik Engine.IO session
- `AsyncRedisManager` + relay `ws:relay:events` pour broadcast cross-pod
- Dedup cross-pod : Redis SET `dedup:{user_id}:{room}:{event_id}` TTL 90s
- Kill switch par replica

## Gate scaling staging

2 replicas, mixed population, pas de spike dedup, kill switch OK.
