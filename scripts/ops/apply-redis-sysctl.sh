#!/usr/bin/env bash
# Applique vm.overcommit_memory=1 (recommandation Redis) sur l'hôte.
# Usage : sudo ./scripts/ops/apply-redis-sysctl.sh
set -euo pipefail

sysctl -w vm.overcommit_memory=1
CONF="/etc/sysctl.conf"
if ! grep -q '^vm.overcommit_memory' "${CONF}" 2>/dev/null; then
  echo 'vm.overcommit_memory = 1' >>"${CONF}"
  echo "Persisté dans ${CONF}"
else
  echo "Déjà présent dans ${CONF}"
fi
sysctl vm.overcommit_memory
