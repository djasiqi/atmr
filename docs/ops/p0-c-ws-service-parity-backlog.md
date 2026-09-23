# P0-C — backlog de parité ws-service et libellé RATE_LIMIT

Ces écarts bloquent une bascule générale de `/socket.io` vers ws-service.
Ils ne bloquent pas le canary company qui sert à prouver le long-polling.

Aucun de ces points n'est corrigé dans cette passe.

## Blocker canary authentifié (2026-09-23)

| Écart | Legacy | ws-service | Effet |
|---|---|---|---|
| JWT `aud` | Access tokens avec `aud=atmr-api` ou `aud=atmr-mobile-enterprise`. Login mobile entreprise → `atmr-mobile-enterprise`. | **Code local corrigé** (`jwt_auth.py` : `audience=[atmr-api, atmr-mobile-enterprise]` + `type=access`). **Pas encore déployé en prod.** | Tant que l'image prod n'embarque pas ce patch, un JWT réel est refusé (`InvalidAudienceError`). |

## Écarts

| Écart | Legacy | ws-service | Effet si bascule générale |
|---|---|---|---|
| Chat | `team_chat_message`, `team_chat_typing`, `conversation_message` sont émis par `backend/sockets/chat.py` via `emit()` Flask. Ce chemin ne publie pas `ws:relay:events`. | Aucun handler entrant pour ces événements. | Le canary company ne reçoit pas le chat, et un client sur ws-service ne peut pas en envoyer. |
| `booking_message` | `socketio.emit` direct (`backend/services/events/institution_events.py`, `backend/routes/booking_messages.py`), hors `_safe_emit`. | Pas de relay de cette voie. | Messages de réservation absents du client canary. |
| `institution_offer_updated` | `socketio.emit` direct dans `backend/application/institutions/send_transport_request.py`. | Pas relayé. | Offres institution absentes. |
| `new_company_notification` | `socketio.emit` direct dans `persist_company_notification`. | Pas relayé. | Notifications company absentes. |
| GPS batch ACK | `driver_location_batch` répond `driver_location_batch_ack` (`backend/sockets/chat.py`). | Le handler existe, mais la réponse d'échec est `driver_location_nack`. Pas de `driver_location_batch_ack`. | Un chauffeur basculé ne reçoit pas l'ACK que l'app attend. |
| Nom `dispatch_assignment` | Les noms émis sont `dispatch_assignment_created`, `dispatch_assignment_updated`, `dispatch_assignment_cancelled`. | Le relay reprendrait ces noms. | L'app écoute `dispatch_assignment`, qui n'est pas le nom émis, déjà côté legacy. |
| Nom `delay_invalidated` | Le nom émis est `delay_live_invalidate`. | Le relay reprendrait ce nom. | L'app écoute `delay_invalidated`. L'écart existe déjà sur le legacy. |

## RATE_LIMIT affiché pendant l'incident Invalid session

Correctif UI non fait. À traiter après stabilisation du transport.

L'écran company (`mobile/unified-app/app/(app)/(company)/rides.tsx`) affiche `realtimeStatus.lastError` tel quel, sous le titre « Temps réel indisponible ».

`lastError` est posé par `companyRealtimeBridge.setTransportStatus`. Sans nouveau message, la valeur précédente reste. Le handler `disconnect` appelle `setTransportStatus("reconnecting")` sans message.

La chaîne exacte `RATE_LIMIT` est produite par `SocketConnectionRefusedError("RATE_LIMIT")` dans `backend/sockets/chat.py`, quand le limiteur de connexion refuse. Le bridge company ne transforme pas `Invalid session` en `RATE_LIMIT`. Le libellé français de ce code est dans le frontend web `frontend/src/services/socketStatusReasons.js`, pas dans l'écran téléphone.

Pendant la rafale du 23.09.2026 à 19:45 UTC, les logs backend ne contiennent pas `RATE_LIMIT` ni `socket_rate_limit_exceeded`. Les requêtes observées sont un handshake polling 200 puis HTTP 400 `Invalid session`. Le texte `RATE_LIMIT` affiché est donc un `lastError` antérieur, conservé, et non l'erreur du moment.
