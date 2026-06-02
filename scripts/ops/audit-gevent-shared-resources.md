# Audit gevent / ressources partagees (P0c)

## Contexte

Les erreurs `ConcurrentObjectUseError` et `Worker exited with code 1` sous Gunicorn gevent
peuvent provenir de connexions partagees entre greenlets.

## Etat actuel du code

### Redis (`backend/ext.py`)

- `redis_client = redis.Redis.from_url(...)` : pool de connexions redis-py (thread-safe par connexion du pool).
- Sous gevent, preferer `redis.connection.ConnectionPool` avec `max_connections` explicite.
- Ne pas partager une connexion Redis unique entre handlers concurrents.

### SQLAlchemy (`backend/ext.py`)

- `db = SQLAlchemy()` : scoped session Flask par requete HTTP (OK pour requetes WSGI).
- Workers Celery : session nettoyee dans `run_dispatch_task` via `db.session.rollback()`.

### Celery enqueue (`backend/services/unified_dispatch/core/queue.py`)

- **Corrige P0b** : suppression du monkey-patch global `celery_app._connection_for_write`.
- Mesurer la chute de `ConcurrentObjectUseError` apres deploiement P0b.

### Socket.IO (`backend/ext.py`)

- `SocketIO` avec `message_queue=REDIS_URL` pour multi-workers (OK).
- `StopIteration` sur upgrade WebSocket : bruit Sentry filtre en P3.

## Actions post-deploy P0b

1. Correler Sentry `ConcurrentObjectUseError` avec logs `WORKER TIMEOUT` / `WORKER ABORT` (`gunicorn.conf.py`).
2. Si reliquat > 0 apres 48h : activer `GEVENT_SUPPORT=True` sur redis-py ou migrer vers pool gevent-safe.
3. Verifier qu'aucun client DB/Redis n'est cree au niveau module et reutilise entre greenlets.

## KPI

- Sentry : issues PYTHON-FLASK-9N, 9M, 9J
- Logs Gunicorn : `WORKER TIMEOUT pid=`, `WORKER ABORT pid=`
