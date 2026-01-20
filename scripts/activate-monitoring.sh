#!/bin/bash
# Script pour activer Prometheus, Grafana et Alertmanager en production
# Usage: ssh user@host 'bash -s' < scripts/activate-monitoring.sh
# Ou: exécuter directement sur le serveur: bash scripts/activate-monitoring.sh

set -o errexit -o nounset -o pipefail

cd /srv/atmr || { echo "❌ Répertoire /srv/atmr non trouvé"; exit 1; }

echo "📊 Activation des services de monitoring (Prometheus, Grafana, Alertmanager)..."

# Vérifier que les fichiers nécessaires existent
if [ ! -f "docker-compose.monitoring.yml" ]; then
  echo "❌ docker-compose.monitoring.yml non trouvé"
  exit 1
fi

if [ ! -d "monitoring" ]; then
  echo "❌ Dossier monitoring/ non trouvé"
  exit 1
fi

# Préparer les fichiers nécessaires
echo "🔧 Préparation des fichiers..."
[ -f "monitoring/alertmanager/docker-entrypoint.sh" ] && chmod +x monitoring/alertmanager/docker-entrypoint.sh || true

# Construire l'image Alertmanager si nécessaire
if [ -f "monitoring/alertmanager/Dockerfile" ]; then
  echo "🔨 Construction de l'image Alertmanager si nécessaire..."
  docker compose -f docker-compose.monitoring.yml build alertmanager || echo "⚠️  Build Alertmanager échoué (peut être ignoré si l'image existe déjà)"
fi

# Démarrer les services de monitoring
echo "🔄 Démarrage des services de monitoring (Prometheus, Grafana, Alertmanager)..."
if ! docker compose -f docker-compose.monitoring.yml up -d --remove-orphans; then
  echo "❌ Échec du démarrage du monitoring"
  echo "📋 Logs du monitoring:"
  docker compose -f docker-compose.monitoring.yml logs --tail=50 || true
  exit 1
fi

echo "✅ Commandes de démarrage du monitoring exécutées"

# Attendre que les services démarrent
echo "⏳ Attente du démarrage des services (15 secondes)..."
sleep 15

# Vérifier le statut des services
echo "🔍 Vérification du statut des services..."
MONITORING_OK=true
for service in prometheus grafana alertmanager; do
  SERVICE_STATUS=$(docker compose -f docker-compose.monitoring.yml ps "$service" --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
  if [ "$SERVICE_STATUS" != "running" ]; then
    echo "⚠️  Service $service n'est pas en cours d'exécution (status: $SERVICE_STATUS)"
    MONITORING_OK=false
    # Afficher les logs du service en erreur
    echo "📋 Logs de $service:"
    docker compose -f docker-compose.monitoring.yml logs "$service" --tail=20 || true
  else
    echo "✅ Service $service démarré (status: $SERVICE_STATUS)"
  fi
done

if [ "$MONITORING_OK" = "false" ]; then
  echo "❌ Certains services de monitoring n'ont pas démarré correctement"
  echo "📋 État de tous les services:"
  docker compose -f docker-compose.monitoring.yml ps || true
  exit 1
fi

# Vérifier les healthchecks
echo "🏥 Vérification des healthchecks..."
for service in prometheus grafana alertmanager; do
  echo "  - Vérification $service..."
  for i in {1..10}; do
    HEALTH=$(docker inspect --format='{{.State.Health.Status}}' "atmr-$service" 2>/dev/null || echo "none")
    if [ "$HEALTH" = "healthy" ]; then
      echo "    ✅ $service est healthy"
      break
    elif [ "$i" -eq 10 ]; then
      echo "    ⚠️  $service healthcheck timeout (status: $HEALTH)"
    else
      sleep 2
    fi
  done
done

# Afficher les URLs d'accès
echo ""
echo "✅ Services de monitoring activés avec succès!"
echo ""
echo "📊 URLs d'accès:"
echo "  - Prometheus: https://prometheus.lirie.ch"
echo "  - Grafana: https://grafana.lirie.ch"
echo "  - Alertmanager: https://alertmanager.lirie.ch"
echo ""
echo "📋 Statut des services:"
docker compose -f docker-compose.monitoring.yml ps
