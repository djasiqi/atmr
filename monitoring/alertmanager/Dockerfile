FROM prom/alertmanager:v0.26.0

# Le script d'entrypoint utilise sed pour la substitution (disponible sur toutes les distributions)
# Pas besoin d'installer gettext/envsubst car sed fonctionne parfaitement
USER root

# Copier le script d'entrypoint
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Revenir à l'utilisateur non-root
USER nobody

ENTRYPOINT ["/docker-entrypoint.sh"]

