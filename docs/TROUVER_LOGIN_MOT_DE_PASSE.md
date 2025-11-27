# 🔐 Comment retrouver votre login et mot de passe ATMR

## Méthode 1 : Lister les utilisateurs existants

Connectez-vous au serveur de production et exécutez :

```bash
# Se connecter au conteneur backend
docker exec -it atmr-backend python scripts/manage_users.py list
```

Cela affichera tous les utilisateurs avec leur ID, username, email et rôle.

## Méthode 2 : Créer un utilisateur admin

Si aucun utilisateur n'existe, créez-en un :

```bash
# Se connecter au conteneur backend
docker exec -it atmr-backend python scripts/manage_users.py create-admin \
  --username admin \
  --email admin@example.com \
  --password VotreMotDePasse123 \
  --role ADMIN
```

**Note** : Remplacez `admin@example.com` et `VotreMotDePasse123` par vos valeurs.

## Méthode 3 : Réinitialiser le mot de passe d'un utilisateur existant

### Par username :

```bash
docker exec -it atmr-backend python scripts/manage_users.py reset-password \
  --username admin \
  --new-password NouveauMotDePasse123
```

### Par ID utilisateur :

```bash
docker exec -it atmr-backend python scripts/manage_users.py reset-password \
  --user-id 1 \
  --new-password NouveauMotDePasse123
```

## Méthode 4 : Via la base de données PostgreSQL (avancé)

Si vous préférez interroger directement la base de données :

```bash
# Se connecter au conteneur PostgreSQL
docker exec -it atmr-postgres psql -U atmr -d atmr

# Lister les utilisateurs
SELECT id, username, email, role FROM "user";

# Pour réinitialiser un mot de passe, vous devez hasher le mot de passe avec werkzeug
# Utilisez plutôt le script Python ci-dessus
```

## Exemple complet

```bash
# 1. Lister les utilisateurs existants
docker exec -it atmr-backend python scripts/manage_users.py list

# 2. Si aucun utilisateur n'existe, en créer un
docker exec -it atmr-backend python scripts/manage_users.py create-admin \
  --username admin \
  --email admin@lirie.ch \
  --password MonMotDePasseSecurise123 \
  --role ADMIN

# 3. Se connecter avec ces identifiants sur https://lirie.ch
```

## Rôles disponibles

- `ADMIN` : Administrateur système
- `CLIENT` : Client (utilisateur final)
- `DRIVER` : Chauffeur
- `COMPANY` : Compagnie de transport

## Sécurité

⚠️ **Important** : Après avoir créé ou réinitialisé un mot de passe, changez-le immédiatement via l'interface web pour des raisons de sécurité.
