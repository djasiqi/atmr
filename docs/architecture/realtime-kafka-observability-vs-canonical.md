# Temps réel : observabilité Kafka vs canon Redis

## I2 — Ce n’est pas un bug de code

Le contrat dans `backend/services/realtime/socketio.py` (`fanout_driver_location_update`) est volontaire :

- **`driver_location_update`** : position pour la carte (canon **ou** observabilité).
- **`driver_live_state_update`** : **uniquement** pour `accepted_canonical` — pas pour `accepted_observability_only`.

Le consumer Kafka `processed_fanout_consumer` émet en `accepted_observability_only` : la carte reçoit la géométrie, mais **pas** un nouvel état métier « canon » (couleur / mission / disponibilité) tant que le flux canon (HTTP/socket avec arbitrage Redis) ne l’a pas décidé.

Toute évolution (ex. « Kafka devient canon ») est une **décision produit / architecture**, pas un correctif rapide.

## I5 — ACK réel mobile (file d’attente)

Avant d’activer `EXPO_PUBLIC_ENABLE_TRACKING_REAL_ACK_SEMANTICS=1` en production :

1. Vérifier côté backend que le flux socket (ou HTTP) utilisé par `driverTrackingQueue` renvoie bien un **ACK de séquence** (`ack_last_sequence_id` ou équivalent documenté).
2. Activer d’abord en **staging** avec monitoring de la taille de file, des rejouages et des pertes après coupure réseau.
3. Ne pas activer en prod sans validation métrique.

## Références code

- Fanout : `backend/services/realtime/socketio.py`
- Fanout Kafka processed : `backend/services/tracking/processed_fanout_consumer.py`
- Queue mobile : `mobile/unified-app/src/features/driver/services/driverTrackingQueue.ts`
