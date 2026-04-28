from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import logging
import os
import time
from contextlib import suppress
from typing import Any

import jwt
import redis.asyncio as redis
import socketio
from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from aiokafka import AIOKafkaConsumer
except ImportError:  # pragma: no cover
    AIOKafkaConsumer = None

logger = logging.getLogger("ws-service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
AUTH_CACHE_TTL_SECONDS = int(os.getenv("WS_AUTH_CACHE_TTL_SECONDS", "60"))
PRESENCE_TTL_SECONDS = int(os.getenv("WS_PRESENCE_TTL_SECONDS", "90"))
AUTH_CACHE_PREFIX = os.getenv("WS_AUTH_CACHE_PREFIX", "ws:auth:token:")
PRESENCE_SID_PREFIX = os.getenv("WS_PRESENCE_SID_PREFIX", "ws:presence:sid:")
PRESENCE_DRIVER_PREFIX = os.getenv("WS_PRESENCE_DRIVER_PREFIX", "ws:presence:driver:")
PRESENCE_DRIVER_SESSIONS_PREFIX = os.getenv(
    "WS_PRESENCE_DRIVER_SESSIONS_PREFIX", "ws:presence:driver_sids:"
)
WS_AUTHORITY_SOURCE = os.getenv("WS_AUTHORITY_SOURCE", "ws-service")
WS_VERSION = os.getenv("WS_VERSION", "v1")
REDIS_URL = os.getenv("REDIS_CLUSTER_URL", "redis://redis:6379/0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Redis Cluster mode (Phase 4).
# Lorsque REDIS_CLUSTER_MODE=true, le ws-service utilise redis.asyncio.RedisCluster
# pour les opérations de présence/auth (hash-slot aware) et continue d'utiliser
# socketio.AsyncRedisManager via le premier nœud du cluster pour le pub/sub cross-pod
# (Redis Cluster redirige pub/sub vers le bon nœud automatiquement).
REDIS_CLUSTER_MODE = os.getenv("REDIS_CLUSTER_MODE", "false").lower() == "true"
# URL du premier nœud Cluster pour initialisation (discovery automatique des autres).
# En mode single: identique à REDIS_CLUSTER_URL.
# En mode cluster: redis://redis-node-1:7001 (nœud de bootstrap suffisant).
REDIS_CLUSTER_STARTUP_NODE_URL = os.getenv(
    "REDIS_CLUSTER_STARTUP_NODE_URL", REDIS_URL
)
ALLOW_IF_REDIS_PRESENCE_DOWN = (
    os.getenv("WS_ALLOW_IF_REDIS_PRESENCE_DOWN", "true").lower() == "true"
)
ALLOW_IF_REDIS_AUTH_DOWN = (
    os.getenv("WS_ALLOW_IF_REDIS_AUTH_DOWN", "false").lower() == "true"
)
KAFKA_CONSUMER_ENABLED = os.getenv("WS_KAFKA_CONSUMER_ENABLED", "true").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED", "driver.location.processed"
)
KAFKA_TOPIC_MISSION_EVENTS = os.getenv("KAFKA_TOPIC_MISSION_EVENTS", "mission.events")
WS_KAFKA_ENABLE_MISSION_EVENTS = (
    os.getenv("WS_KAFKA_ENABLE_MISSION_EVENTS", "false").lower() == "true"
)
WS_KAFKA_GROUP_ID = os.getenv("WS_KAFKA_GROUP_ID", "ws-service-shared")
WS_KAFKA_AUTO_OFFSET_RESET = os.getenv("WS_KAFKA_AUTO_OFFSET_RESET", "latest")
WS_KAFKA_SESSION_TIMEOUT_MS = int(os.getenv("WS_KAFKA_SESSION_TIMEOUT_MS", "30000"))
WS_KAFKA_HEARTBEAT_INTERVAL_MS = int(os.getenv("WS_KAFKA_HEARTBEAT_INTERVAL_MS", "10000"))
WS_KAFKA_REBALANCE_TIMEOUT_MS = int(os.getenv("WS_KAFKA_REBALANCE_TIMEOUT_MS", "60000"))
WS_KAFKA_SEEK_TO_END_ON_START = (
    os.getenv("WS_KAFKA_SEEK_TO_END_ON_START", "true").lower() == "true"
)
WS_KAFKA_SECURITY_PROTOCOL = os.getenv("WS_KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
WS_KAFKA_SASL_MECHANISM = os.getenv("WS_KAFKA_SASL_MECHANISM", "")
WS_KAFKA_SASL_USERNAME = os.getenv("WS_KAFKA_SASL_USERNAME", "")
WS_KAFKA_SASL_PASSWORD = os.getenv("WS_KAFKA_SASL_PASSWORD", "")
WS_KAFKA_SSL_CAFILE = os.getenv("WS_KAFKA_SSL_CAFILE", "")
WS_KAFKA_SSL_CERTFILE = os.getenv("WS_KAFKA_SSL_CERTFILE", "")
WS_KAFKA_SSL_KEYFILE = os.getenv("WS_KAFKA_SSL_KEYFILE", "")
WS_RELAY_CHANNEL = os.getenv("WS_RELAY_CHANNEL", "ws:relay:events")
POD_ID = os.getenv("POD_ID", os.getenv("HOSTNAME", "local"))

redis_client: redis.Redis | None = None
redis_manager: socketio.AsyncRedisManager | None = None
kafka_consumer_task: asyncio.Task[Any] | None = None
relay_listener_task: asyncio.Task[Any] | None = None


def _build_redis_connection_url() -> str:
    """Construit l'URL Redis avec mot de passe si fourni (mode single)."""
    if not REDIS_PASSWORD:
        return REDIS_URL
    if "@" in REDIS_URL:
        return REDIS_URL
    return REDIS_URL.replace("redis://", f"redis://:{REDIS_PASSWORD}@", 1)


def _build_redis_startup_node_url() -> str:
    """Construit l'URL du nœud de bootstrap Redis Cluster."""
    url = REDIS_CLUSTER_STARTUP_NODE_URL
    if not REDIS_PASSWORD:
        return url
    if "@" in url:
        return url
    return url.replace("redis://", f"redis://:{REDIS_PASSWORD}@", 1)


async def _create_redis_client() -> redis.Redis:
    """Instancie le client Redis adapté au mode (single vs cluster).

    En mode cluster (REDIS_CLUSTER_MODE=true) :
      - Retourne un redis.asyncio.RedisCluster initialisé sur le nœud de
        bootstrap ; la découverte automatique gère les autres nœuds.
      - La sémantique est identique à redis.Redis pour les opérations GET/SET/
        EXPIRE/PUBLISH utilisées par le ws-service (les pipelines multi-clés
        doivent respecter la co-location hash-slot, ce qui est le cas ici
        car les clés de présence utilisent toutes {driver_id} comme hash tag).

    En mode single (défaut) :
      - Comportement identique à l'implémentation précédente.
    """
    if REDIS_CLUSTER_MODE:
        try:
            # redis.asyncio.RedisCluster est disponible dans redis-py >= 4.1
            cluster_client = redis.asyncio.RedisCluster.from_url(
                _build_redis_startup_node_url(),
                decode_responses=True,
                skip_full_coverage_check=True,  # Autorise cluster sans tous les slots couverts
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            logger.info("redis cluster client created (startup node: %s)", REDIS_CLUSTER_STARTUP_NODE_URL)
            return cluster_client
        except AttributeError:
            # Fallback si la version de redis-py ne supporte pas RedisCluster.from_url
            logger.warning(
                "redis.asyncio.RedisCluster.from_url not available, falling back to single-node client. Upgrade redis-py >= 4.1."
            )

    # Mode single (ou fallback cluster)
    return redis.from_url(
        _build_redis_connection_url(),
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=os.getenv("WS_CORS_ALLOWED_ORIGINS", "*").split(","),
    logger=False,
    engineio_logger=False,
    ping_interval=int(os.getenv("WS_PING_INTERVAL_SECONDS", "25")),
    ping_timeout=int(os.getenv("WS_PING_TIMEOUT_SECONDS", "15")),
)
fastapi_app = FastAPI(title="ATMR ws-service", version=WS_VERSION)
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)


def _token_cache_key(token: str) -> str:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"{AUTH_CACHE_PREFIX}{token_hash}"


def _presence_sid_key(sid: str) -> str:
    return f"{PRESENCE_SID_PREFIX}{sid}"


def _presence_driver_key(driver_id: int) -> str:
    return f"{PRESENCE_DRIVER_PREFIX}{driver_id}"


def _presence_driver_sessions_key(driver_id: int) -> str:
    return f"{PRESENCE_DRIVER_SESSIONS_PREFIX}{driver_id}"


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _resolve_redis_call(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _driver_room(driver_id: int) -> str:
    return f"driver:{driver_id}"


def _has_local_room_members(room: str, namespace: str = "/") -> bool:
    try:
        namespace_rooms = sio.manager.rooms.get(namespace, {})
        members = namespace_rooms.get(room)
        return bool(members)
    except Exception:
        return False


async def _relay_event_if_needed(event: dict[str, Any], room: str) -> str:
    if _has_local_room_members(room):
        await sio.emit(event["type"], event["payload"], to=room)
        return "local"

    if redis_client is None:
        return "none"

    payload = {
        "source_pod": POD_ID,
        "room": room,
        "event_type": event["type"],
        "payload": event["payload"],
        "ts": _now_ms(),
    }
    await redis_client.publish(WS_RELAY_CHANNEL, json.dumps(payload))
    return "redis"


async def _consume_kafka_events() -> None:
    if not KAFKA_CONSUMER_ENABLED:
        logger.info("kafka consumer disabled by env")
        return
    if AIOKafkaConsumer is None:
        logger.error("aiokafka not installed; disabling ws kafka consumer")
        return

    topics = [KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED]
    if WS_KAFKA_ENABLE_MISSION_EVENTS:
        topics.append(KAFKA_TOPIC_MISSION_EVENTS)
    else:
        logger.info("mission.events consumer disabled by env")

    consumer_kwargs: dict[str, Any] = {
        "bootstrap_servers": [
            server.strip() for server in KAFKA_BOOTSTRAP_SERVERS.split(",") if server.strip()
        ],
        "group_id": WS_KAFKA_GROUP_ID,
        "enable_auto_commit": True,
        "auto_offset_reset": WS_KAFKA_AUTO_OFFSET_RESET,
        "session_timeout_ms": WS_KAFKA_SESSION_TIMEOUT_MS,
        "heartbeat_interval_ms": WS_KAFKA_HEARTBEAT_INTERVAL_MS,
        "rebalance_timeout_ms": WS_KAFKA_REBALANCE_TIMEOUT_MS,
        "security_protocol": WS_KAFKA_SECURITY_PROTOCOL,
    }
    if WS_KAFKA_SASL_MECHANISM:
        consumer_kwargs["sasl_mechanism"] = WS_KAFKA_SASL_MECHANISM
    if WS_KAFKA_SASL_USERNAME:
        consumer_kwargs["sasl_plain_username"] = WS_KAFKA_SASL_USERNAME
    if WS_KAFKA_SASL_PASSWORD:
        consumer_kwargs["sasl_plain_password"] = WS_KAFKA_SASL_PASSWORD
    if WS_KAFKA_SSL_CAFILE:
        consumer_kwargs["ssl_cafile"] = WS_KAFKA_SSL_CAFILE
    if WS_KAFKA_SSL_CERTFILE:
        consumer_kwargs["ssl_certfile"] = WS_KAFKA_SSL_CERTFILE
    if WS_KAFKA_SSL_KEYFILE:
        consumer_kwargs["ssl_keyfile"] = WS_KAFKA_SSL_KEYFILE

    consumer = AIOKafkaConsumer(*topics, **consumer_kwargs)
    try:
        await consumer.start()
        logger.info("kafka consumer started topics=%s group_id=%s pod=%s", topics, WS_KAFKA_GROUP_ID, POD_ID)
        if WS_KAFKA_SEEK_TO_END_ON_START:
            await consumer.seek_to_end()
            logger.info(
                "kafka consumer seek_to_end applied group_id=%s pod=%s assignments=%s",
                WS_KAFKA_GROUP_ID,
                POD_ID,
                len(consumer.assignment()),
            )
        async for msg in consumer:
            raw_value = msg.value
            if not isinstance(raw_value, (bytes, bytearray)):
                continue
            try:
                decoded_value = raw_value.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning(
                    "kafka event skipped topic=%s partition=%s offset=%s reason=invalid_encoding",
                    msg.topic,
                    msg.partition,
                    msg.offset,
                )
                continue
            try:
                value = json.loads(decoded_value)
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(decoded_value)
                except (ValueError, SyntaxError):
                    logger.warning(
                        "kafka event skipped topic=%s partition=%s offset=%s reason=invalid_json",
                        msg.topic,
                        msg.partition,
                        msg.offset,
                    )
                    continue
            if not isinstance(value, dict):
                continue

            driver_id_obj = value.get("driver_id")
            if not isinstance(driver_id_obj, int):
                continue

            event_type_obj = value.get("type") or value.get("event_type") or msg.topic
            event_payload_obj = value.get("payload", value)
            if not isinstance(event_type_obj, str) or not isinstance(event_payload_obj, dict):
                continue
            event_id_obj = event_payload_obj.get("event_id") if isinstance(event_payload_obj.get("event_id"), str) else "n/a"

            room = _driver_room(driver_id_obj)
            relay_mode = await _relay_event_if_needed(
                {
                    "type": event_type_obj,
                    "payload": event_payload_obj,
                },
                room,
            )
            logger.info(
                "kafka event processed topic=%s partition=%s offset=%s driver_id=%s event_type=%s event_id=%s relay_mode=%s room=%s",
                msg.topic,
                msg.partition,
                msg.offset,
                driver_id_obj,
                event_type_obj,
                event_id_obj,
                relay_mode,
                room,
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("kafka consumer loop failed")
    finally:
        await consumer.stop()


async def _listen_relay_events() -> None:
    if redis_client is None:
        return

    pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
    await pubsub.subscribe(WS_RELAY_CHANNEL)
    logger.info("relay listener subscribed channel=%s pod=%s", WS_RELAY_CHANNEL, POD_ID)
    try:
        while True:
            message = await pubsub.get_message(timeout=1.0)
            if not message:
                await asyncio.sleep(0.05)
                continue

            data = message.get("data")
            if not isinstance(data, str):
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("source_pod") == POD_ID:
                continue

            room_obj = payload.get("room")
            event_type_obj = payload.get("event_type")
            event_payload_obj = payload.get("payload")
            if (
                not isinstance(room_obj, str)
                or not isinstance(event_type_obj, str)
                or not isinstance(event_payload_obj, dict)
            ):
                continue
            if not _has_local_room_members(room_obj):
                continue
            await sio.emit(event_type_obj, event_payload_obj, to=room_obj)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("relay listener failed")
    finally:
        with suppress(Exception):
            await pubsub.unsubscribe(WS_RELAY_CHANNEL)
            await pubsub.aclose()


async def _decode_token(token: str) -> dict[str, Any] | None:
    if not JWT_SECRET_KEY:
        logger.error("JWT_SECRET_KEY is missing")
        return None

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None

    if not isinstance(payload, dict):
        return None
    return payload


async def _get_auth_payload(token: str) -> dict[str, Any] | None:
    if redis_client is None:
        return await _decode_token(token)

    key = _token_cache_key(token)
    try:
        cached = await redis_client.get(key)
    except Exception:
        logger.exception("auth cache read failure")
        if ALLOW_IF_REDIS_AUTH_DOWN:
            return await _decode_token(token)
        return None

    if cached:
        try:
            decoded_cached = json.loads(cached)
            if isinstance(decoded_cached, dict):
                return decoded_cached
        except json.JSONDecodeError:
            logger.warning("auth cache contained invalid json")

    payload = await _decode_token(token)
    if payload is None:
        return None

    try:
        await redis_client.setex(key, AUTH_CACHE_TTL_SECONDS, json.dumps(payload))
    except Exception:
        logger.exception("auth cache write failure")
    return payload


async def _register_presence(
    *,
    sid: str,
    user_id: int,
    role: str,
    company_id: int | None,
    driver_id: int | None,
) -> None:
    if redis_client is None:
        if ALLOW_IF_REDIS_PRESENCE_DOWN:
            return
        raise ConnectionError("redis_presence_unavailable")

    sid_key = _presence_sid_key(sid)
    payload = {
        "sid": sid,
        "user_id": user_id,
        "role": role,
        "company_id": company_id,
        "driver_id": driver_id,
        "authority": WS_AUTHORITY_SOURCE,
        "last_seen_ms": _now_ms(),
    }
    serialized = json.dumps(payload)

    await redis_client.setex(sid_key, PRESENCE_TTL_SECONDS, serialized)

    if driver_id is None:
        return

    canonical_key = _presence_driver_key(driver_id)
    sessions_key = _presence_driver_sessions_key(driver_id)
    previous_raw = await redis_client.get(canonical_key)
    previous_sid: str | None = None

    if previous_raw:
        try:
            previous_payload = json.loads(previous_raw)
            if isinstance(previous_payload, dict):
                previous_sid_obj = previous_payload.get("sid")
                if isinstance(previous_sid_obj, str):
                    previous_sid = previous_sid_obj
        except json.JSONDecodeError:
            logger.warning("driver canonical presence had invalid payload")

    await redis_client.setex(canonical_key, PRESENCE_TTL_SECONDS, serialized)
    await _resolve_redis_call(redis_client.sadd(sessions_key, sid))
    await _resolve_redis_call(redis_client.expire(sessions_key, PRESENCE_TTL_SECONDS))

    if previous_sid and previous_sid != sid:
        await _resolve_redis_call(redis_client.srem(sessions_key, previous_sid))
        await _resolve_redis_call(redis_client.delete(_presence_sid_key(previous_sid)))
        logger.info("presence arbitration: driver_id=%s old_sid=%s new_sid=%s", driver_id, previous_sid, sid)


async def _refresh_presence(sid: str) -> None:
    if redis_client is None:
        return

    sid_key = _presence_sid_key(sid)
    sid_payload_raw = await redis_client.get(sid_key)
    if not sid_payload_raw:
        return

    await redis_client.expire(sid_key, PRESENCE_TTL_SECONDS)

    with suppress(json.JSONDecodeError):
        payload = json.loads(sid_payload_raw)
        if isinstance(payload, dict):
            driver_id_obj = payload.get("driver_id")
            if isinstance(driver_id_obj, int):
                await redis_client.expire(
                    _presence_driver_key(driver_id_obj), PRESENCE_TTL_SECONDS
                )
                await redis_client.expire(
                    _presence_driver_sessions_key(driver_id_obj), PRESENCE_TTL_SECONDS
                )


async def _remove_presence(sid: str) -> None:
    if redis_client is None:
        return

    sid_key = _presence_sid_key(sid)
    sid_payload_raw = await redis_client.get(sid_key)
    await redis_client.delete(sid_key)
    if not sid_payload_raw:
        return

    with suppress(json.JSONDecodeError):
        payload = json.loads(sid_payload_raw)
        if not isinstance(payload, dict):
            return
        driver_id_obj = payload.get("driver_id")
        if not isinstance(driver_id_obj, int):
            return

        sessions_key = _presence_driver_sessions_key(driver_id_obj)
        canonical_key = _presence_driver_key(driver_id_obj)
        await _resolve_redis_call(redis_client.srem(sessions_key, sid))
        remaining = await _resolve_redis_call(redis_client.scard(sessions_key))
        if remaining <= 0:
            await redis_client.delete(sessions_key)

        current_raw = await redis_client.get(canonical_key)
        if not current_raw:
            return
        with suppress(json.JSONDecodeError):
            current_payload = json.loads(current_raw)
            if isinstance(current_payload, dict) and current_payload.get("sid") == sid:
                await redis_client.delete(canonical_key)


async def _extract_token(auth: dict[str, Any] | None, environ: dict[str, Any]) -> str | None:
    if auth and isinstance(auth.get("token"), str):
        return auth["token"]

    header_value = environ.get("HTTP_AUTHORIZATION")
    if isinstance(header_value, str) and header_value.lower().startswith("bearer "):
        return header_value[7:]
    return None


@sio.event
async def connect(sid: str, environ: dict[str, Any], auth: dict[str, Any] | None) -> bool:
    token = await _extract_token(auth, environ)
    if not token:
        logger.info("socket reject: missing token sid=%s", sid)
        return False

    payload = await _get_auth_payload(token)
    if payload is None:
        logger.info("socket reject: invalid token sid=%s", sid)
        return False

    user_id_obj = payload.get("sub") or payload.get("user_id")
    role_obj = payload.get("role")
    company_id_obj = payload.get("company_id")
    driver_id_obj = payload.get("driver_id")

    if not isinstance(user_id_obj, int) or not isinstance(role_obj, str):
        logger.info("socket reject: invalid claims sid=%s", sid)
        return False

    company_id = company_id_obj if isinstance(company_id_obj, int) else None
    driver_id = driver_id_obj if isinstance(driver_id_obj, int) else None

    await sio.save_session(
        sid,
        {
            "user_id": user_id_obj,
            "role": role_obj,
            "company_id": company_id,
            "driver_id": driver_id,
        },
    )

    if driver_id is not None:
        await sio.enter_room(sid, f"driver:{driver_id}")
    if company_id is not None:
        await sio.enter_room(sid, f"company:{company_id}")

    try:
        await _register_presence(
            sid=sid,
            user_id=user_id_obj,
            role=role_obj,
            company_id=company_id,
            driver_id=driver_id,
        )
    except ConnectionError:
        logger.warning("socket reject: redis presence unavailable sid=%s", sid)
        return False

    await sio.emit(
        "connection.authority",
        {
            "authority": WS_AUTHORITY_SOURCE,
            "canary": True,
            "version": WS_VERSION,
        },
        to=sid,
    )
    logger.info(
        "socket connected sid=%s user_id=%s role=%s company_id=%s driver_id=%s authority=%s",
        sid,
        user_id_obj,
        role_obj,
        company_id,
        driver_id,
        WS_AUTHORITY_SOURCE,
    )
    return True


@sio.event
async def disconnect(sid: str) -> None:
    await _remove_presence(sid)
    logger.info("socket disconnected sid=%s", sid)


@sio.event
async def presence_heartbeat(sid: str, _: dict[str, Any] | None = None) -> dict[str, Any]:
    await _refresh_presence(sid)
    return {"ok": True, "ts": _now_ms()}


@fastapi_app.get("/health")
async def health() -> JSONResponse:
    redis_up = False
    if redis_client is not None:
        try:
            ping_result = await asyncio.wait_for(
                _resolve_redis_call(redis_client.ping()),
                timeout=1.5,
            )
            redis_up = bool(ping_result)
        except Exception:
            redis_up = False

    return JSONResponse(
        {
            "ok": True,
            "service": "ws-service",
            "authority": WS_AUTHORITY_SOURCE,
            "version": WS_VERSION,
            "redis_up": redis_up,
            "kafka_consumer_enabled": KAFKA_CONSUMER_ENABLED,
            "kafka_consumer_running": kafka_consumer_task is not None and not kafka_consumer_task.done(),
            "degraded_presence": not redis_up and ALLOW_IF_REDIS_PRESENCE_DOWN,
            "degraded_auth": not redis_up and ALLOW_IF_REDIS_AUTH_DOWN,
        }
    )


@fastapi_app.on_event("startup")
async def startup() -> None:
    global redis_client, redis_manager, kafka_consumer_task, relay_listener_task

    redis_client = await _create_redis_client()
    try:
        await _resolve_redis_call(redis_client.ping())
        mode_label = "cluster" if REDIS_CLUSTER_MODE else "single"
        logger.info("redis connected at startup (mode=%s)", mode_label)
    except Exception:
        logger.exception("redis unavailable at startup")
        if not ALLOW_IF_REDIS_PRESENCE_DOWN and not ALLOW_IF_REDIS_AUTH_DOWN:
            raise

    try:
        # socketio.AsyncRedisManager utilise pub/sub Redis.
        # En mode cluster, on pointe sur le nœud de bootstrap : Redis Cluster
        # redirige les commandes SUBSCRIBE/PUBLISH vers le bon slot.
        sio_redis_url = (
            _build_redis_startup_node_url()
            if REDIS_CLUSTER_MODE
            else _build_redis_connection_url()
        )
        redis_manager = socketio.AsyncRedisManager(sio_redis_url)
        sio.manager = redis_manager
    except Exception:
        logger.exception("socket io redis manager unavailable")

    relay_listener_task = asyncio.create_task(_listen_relay_events())
    kafka_consumer_task = asyncio.create_task(_consume_kafka_events())


@fastapi_app.on_event("shutdown")
async def shutdown() -> None:
    global kafka_consumer_task, relay_listener_task
    for task in (kafka_consumer_task, relay_listener_task):
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    kafka_consumer_task = None
    relay_listener_task = None
    if redis_client is not None:
        await redis_client.aclose()
    if redis_manager is not None:
        await asyncio.sleep(0)
