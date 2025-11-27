# backend/Dockerfile.rl
# Dockerfile pour l'environnement Reinforcement Learning / ML lourd
# Cette image contient torch, gymnasium, optuna, etc.
# 
# Usage:
#   docker build -f Dockerfile.rl -t djasiqi/atmr-backend-rl:latest ./backend
#   docker run -it --rm djasiqi/atmr-backend-rl:latest python -m atmr.rl.train

########## Stage 1: Builder (compilation des wheels et dépendances RL) ##########
FROM python:3.11-slim-bookworm AS builder

ENV RL_ENABLED=true
ENV WITH_RL=true

# Variables d'environnement pour optimiser la compilation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_WHEEL_DIR=/wheels \
    PIP_FIND_LINKS=/wheels

WORKDIR /app

# Installation des outils de build (plus complets pour RL)
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libcairo2-dev \
    pkg-config \
    liblcms2-dev \
    libwebp-dev \
    libtiff5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
ARG REQUIREMENTS_HASH
ENV REQUIREMENTS_HASH=${REQUIREMENTS_HASH}
COPY requirements.base.txt requirements-rl.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    echo "📦 Installation de PyTorch CPU-only (plus léger que CUDA)..." && \
    pip wheel --no-cache-dir --wheel-dir /wheels \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch>=2.0.0 torchvision>=0.15.0 && \
    echo "📦 Installation des autres dépendances RL..." && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-rl.txt && \
    (if [ -f requirements-dev.txt ]; then \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-dev.txt; \
    else \
    echo "requirements-dev.txt absent, skip"; \
    fi)

########## Stage 2: Runtime optimisé pour RL ##########
FROM python:3.11-slim-bookworm AS runtime

ENV RL_ENABLED=true
ENV WITH_RL=true

# Variables d'environnement pour la production RL
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    # Optimisations PyTorch pour CPU
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    # Optimisations générales
    PYTHONHASHSEED=random \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# Arguments de build
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Labels pour la traçabilité
LABEL maintainer="ATMR Team" \
    org.opencontainers.image.title="ATMR Backend RL" \
    org.opencontainers.image.description="Backend API pour système de dispatch médical avec RL/ML" \
    org.opencontainers.image.version="${VERSION:-latest}" \
    org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.revision="${VCS_REF}" \
    org.opencontainers.image.vendor="ATMR" \
    org.opencontainers.image.licenses="Proprietary"

# Installation des dépendances runtime
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libpq5 \
    libgomp1 \
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    libcairo2 \
    liblcms2-2 \
    libwebp7 \
    libtiff6 \
    libxml2 \
    libxslt1.1 \
    libffi8 \
    libssl3 \
    libexpat1 \
    libsqlite3-0 \
    libgnutls30 \
    tar \
    gzip \
    curl \
    dumb-init \
    git \
    && apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Installation des wheels depuis le stage builder
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements*.txt ./
RUN python -m pip install --upgrade pip && \
    echo "📦 Installation de PyTorch CPU-only..." && \
    pip install --no-index --find-links=/wheels torch torchvision && \
    echo "📦 Installation des autres dépendances RL..." && \
    pip install --no-index --find-links=/wheels -r requirements-rl.txt && \
    rm -rf /wheels /root/.cache/pip

# Création d'un utilisateur non-root sécurisé
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -u 999 -d /app -s /bin/bash -c "ATMR RL User" appuser && \
    mkdir -p /app/logs /app/data /app/cache /app/data/rl /app/data/ml && \
    chown -R appuser:appgroup /app

# Copie du code source
COPY --chown=appuser:appgroup . /app

# Configuration des permissions sécurisées
RUN chmod -R 755 /app && \
    chmod -R 700 /app/logs /app/data /app/cache && \
    find /app -name "*.py" -exec chmod 644 {} \; && \
    find /app -name "*.sh" -exec chmod 755 {} \;

# Passage à l'utilisateur non-root
USER appuser

# Exposition du port (si utilisé comme API)
EXPOSE 5000

# Configuration des limites de ressources
ENV MEMORY_LIMIT=4G \
    CPU_LIMIT=4

# Script de démarrage
COPY --chown=appuser:appgroup docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Utilisation de dumb-init pour une gestion propre des signaux
ENTRYPOINT ["dumb-init", "--"]
CMD ["/app/docker-entrypoint.sh"]

