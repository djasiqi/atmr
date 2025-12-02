FROM prom/alertmanager:v0.26.0

# Installer gettext pour envsubst (optionnel - le script peut utiliser sed en fallback)
USER root

# Détecter et installer gettext selon la distribution (si possible)
# Si l'installation échoue, le script utilisera sed comme fallback
RUN (command -v apk >/dev/null 2>&1 && apk add --no-cache gettext) || \
    (command -v apt-get >/dev/null 2>&1 && apt-get update && apt-get install -y --no-install-recommends gettext-base && rm -rf /var/lib/apt/lists/*) || \
    (command -v yum >/dev/null 2>&1 && yum install -y gettext && yum clean all) || \
    echo "⚠️  gettext non installé, le script utilisera sed pour la substitution"

# Copier le script d'entrypoint
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Revenir à l'utilisateur non-root
USER nobody

ENTRYPOINT ["/docker-entrypoint.sh"]

