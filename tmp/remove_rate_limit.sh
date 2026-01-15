#!/bin/bash
# Suppression temporaire du rate limiting admin en production

cd /srv/atmr/backend/routes

# Supprimer la ligne contenant le rate limit
grep -v '@limiter.limit("5 per hour")' admin.py > admin.py.tmp && mv admin.py.tmp admin.py

echo "✅ Rate limiting supprimé"
