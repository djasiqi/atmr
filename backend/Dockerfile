# backend/Dockerfile
########## Stage 1: builder (compile wheels, pas de runtime superflu) ##########
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Outils de build uniquement (ne seront pas copiés dans l'image finale)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

############ Stage 2: runtime (léger, non-root, sécurisé) ######################
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    EVENTLET_NO_GREENDNS=yes

WORKDIR /app

# Rendez Postgres optionnel (true par défaut). Construire sans Postgres avec:
#   docker build --build-arg WITH_POSTGRES=false -t atmr-backend .
ARG WITH_POSTGRES=true

# Paquets runtime stricts (pas de toolchain, pas de curl)
RUN apt-get update && \
    if [ "$WITH_POSTGRES" = "true" ]; then apt-get install -y --no-install-recommends libpq5 ca-certificates ; \
    else apt-get install -y --no-install-recommends ca-certificates ; fi && \
    rm -rf /var/lib/apt/lists/*

# Installer les wheels construites au stage builder
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*

# Copier le code
COPY . .

# Créer un utilisateur non-root et donner les droits
RUN useradd -u 10001 -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Healthcheck sans curl (une seule ligne Python)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; \
    try: \
    r=urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3); \
    sys.exit(0 if getattr(r,'status',200)==200 else 1) \
    except Exception: \
    sys.exit(1)"


# Gunicorn en eventlet (Socket.IO + eventlet.monkey_patch)
# Assure-toi que "eventlet" est dans requirements.txt
CMD ["gunicorn", "wsgi:app", \
    "--bind", "0.0.0.0:5000", \
    "--worker-class", "eventlet", \
    "--workers", "1", \
    "--timeout", "120"]