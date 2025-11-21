# Guide de test des migrations avec Docker

## Prérequis

1. Docker et Docker Compose installés
2. Services PostgreSQL et Redis démarrés

## Commandes pour tester les migrations

### 1. Démarrer les services nécessaires (PostgreSQL, Redis)

```bash
# Depuis la racine du projet
docker-compose up -d postgres redis
```

### 2. Vérifier l'état actuel des migrations

```bash
# Exécuter dans le conteneur API
docker-compose run --rm api flask db current
docker-compose run --rm api flask db heads
```

### 3. Appliquer la migration de synchronisation

```bash
# Appliquer toutes les migrations (y compris la nouvelle)
docker-compose run --rm api flask db upgrade heads
```

### 4. Vérifier qu'il n'y a plus de migrations en attente

```bash
# Générer une migration de test (devrait être vide)
docker-compose run --rm api flask db revision --autogenerate -m "test_pending_check"
```

Si cette commande ne génère aucun fichier (ou un fichier vide), c'est que tout est synchronisé.

### 5. Nettoyer la migration de test si elle a été créée

```bash
# Supprimer le fichier de test s'il existe
docker-compose run --rm api sh -c "rm -f migrations/versions/*test_pending_check.py"
```

## Commandes alternatives avec Alembic directement

Si Flask-Migrate pose problème, utilisez Alembic directement :

```bash
# Vérifier l'état
docker-compose run --rm api alembic -c migrations/alembic.ini current
docker-compose run --rm api alembic -c migrations/alembic.ini heads

# Appliquer les migrations
docker-compose run --rm api alembic -c migrations/alembic.ini upgrade heads

# Vérifier les migrations en attente
docker-compose run --rm api alembic -c migrations/alembic.ini revision --autogenerate -m "test"
```

## Test de rollback (optionnel)

```bash
# Obtenir la révision actuelle
CURRENT_REV=$(docker-compose run --rm api flask db current | grep -oE '[a-f0-9]{12,40}' | head -1)

# Obtenir la révision précédente
PREV_REV=$(docker-compose run --rm api flask db history | awk -v current="$CURRENT_REV" 'match($0, /[a-f0-9]{12,40}/, rev) { if (rev[0] == current && prev_rev != "") { print prev_rev; exit } if (rev[0] != "") prev_rev = rev[0] }' | head -1)

# Downgrade
if [ -n "$PREV_REV" ]; then
  docker-compose run --rm api flask db downgrade "$PREV_REV"
  docker-compose run --rm api flask db upgrade heads
fi
```

## Variables d'environnement nécessaires

Assurez-vous que les variables suivantes sont définies dans `backend/.env` ou `docker-compose.yml` :

- `FLASK_APP=backend.wsgi` ou `FLASK_APP=wsgi`
- `DATABASE_URL=postgresql+psycopg://atmr:atmr@postgres:5432/atmr`
- `FLASK_CONFIG=testing` (pour les tests) ou `development` (pour le dev)

## Résultat attendu

Après avoir exécuté `flask db upgrade heads`, la migration `68116559b15d_sync_models_with_db.py` devrait être appliquée.

Après avoir exécuté `flask db revision --autogenerate -m "test"`, aucune migration ne devrait être générée (ou une migration vide).
