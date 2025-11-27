# backend/Dockerfile
########## Stage 1: builder (compile wheels, pas de runtime superflu) ##########
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Outils de build uniquement
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev \
    pkg-config libcairo2-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.base.txt requirements.prod.txt requirements-dev.txt* requirements-otel.txt* ./
RUN python -m pip install --upgrade pip && \
    echo "📦 Installation des dépendances de développement..." && \
    if [ -f requirements-dev.txt ]; then \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-dev.txt; \
    else \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.prod.txt; \
    fi && \
    if [ -f requirements-otel.txt ]; then \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-otel.txt; \
    fi


############ Stage 2: runtime (léger, non-root, sécurisé) ######################
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# Rendez Postgres optionnel (true par défaut). Construire sans Postgres avec:
#   docker build --build-arg WITH_POSTGRES=false -t atmr-backend .
ARG WITH_POSTGRES=true

# Paquets runtime stricts + upgrades sécurité ciblés
# (openssl/libssl, ca-certificates, expat, sqlite, gnutls, tar, gzip)
RUN apt-get update && apt-get upgrade -y && \
    if [ "$WITH_POSTGRES" = "true" ]; then \
    apt-get install -y --no-install-recommends libpq5 ; \
    fi && \
    apt-get install -y --no-install-recommends \
    libcairo2 \
    libgomp1 \
    ca-certificates \
    openssl libssl3 \
    libexpat1 libsqlite3-0 libgnutls30 tar gzip && \
    apt-get upgrade -y --only-upgrade openssl libssl3 ca-certificates libexpat1 libsqlite3-0 libgnutls30 tar gzip && \
    rm -rf /var/lib/apt/lists/*

# Installer les wheels construites au stage builder
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements*.txt ./
RUN python -m pip install --upgrade pip && \
    echo "📦 Installation des dépendances de développement..." && \
    if [ -f requirements-dev.txt ]; then \
    pip install --no-index --find-links=/wheels -r requirements-dev.txt; \
    else \
    pip install --no-index --find-links=/wheels -r requirements.prod.txt; \
    fi && \
    if [ -f requirements-otel.txt ]; then \
    pip install --no-index --find-links=/wheels -r requirements-otel.txt; \
    fi && \
    rm -rf /wheels

# Créer un utilisateur non-root
RUN useradd -u 10001 -m appuser

# Copier le code APRÈS installation des deps (meilleur cache)

COPY . /app
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

# Healthcheck sans curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; \
    r=urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3); \
    sys.exit(0 if getattr(r,'status',200)==200 else 1)"

# Gunicorn en eventlet
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:5000", "--worker-class", "eventlet", "--workers", "1", "--timeout", "120"]