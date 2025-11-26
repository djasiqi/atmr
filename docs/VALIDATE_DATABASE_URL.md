# Guide de validation du secret DATABASE_URL

## Format attendu

Le secret `DATABASE_URL` doit être au format PostgreSQL standard :

```
postgresql://[user]:[password]@[host]:[port]/[database]
```

### Exemple

```
postgresql://atmr_user:mon_mot_de_passe@postgres:5432/atmr_db
```

## Comment vérifier si DATABASE_URL est correct

### Méthode 1 : Construction depuis les variables POSTGRES\_\*

Si vous avez déjà les secrets `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, vous pouvez construire `DATABASE_URL` ainsi :

```bash
# Format de base
postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}

# Exemple concret (remplacez par vos valeurs)
postgresql://atmr_user:mon_mot_de_passe_securise@postgres:5432/atmr_db
```

**⚠️ Important** : Si le mot de passe contient des caractères spéciaux (comme `@`, `#`, `!`, `%`, etc.), ils doivent être échappés en URL. Le code Python le fait automatiquement avec `urllib.parse.quote_plus()`.

### Méthode 2 : Validation avec Python

Créez un script de validation temporaire :

```python
import os
from urllib.parse import quote_plus

# Récupérer les valeurs depuis les secrets GitHub (à adapter)
POSTGRES_USER = "votre_user"
POSTGRES_PASSWORD = "votre_password"
POSTGRES_DB = "votre_db"
POSTGRES_HOST = "postgres"  # ou l'IP du serveur
POSTGRES_PORT = "5432"

# Construire DATABASE_URL avec échappement du mot de passe
password_escaped = quote_plus(POSTGRES_PASSWORD)
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{password_escaped}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

print(f"DATABASE_URL construit: {DATABASE_URL}")

# Valider le format (vérification basique)
if not DATABASE_URL.startswith("postgresql://"):
    print("❌ ERREUR: DATABASE_URL doit commencer par 'postgresql://'")
    exit(1)

if "@" not in DATABASE_URL or ":" not in DATABASE_URL.split("@")[0]:
    print("❌ ERREUR: Format utilisateur:mot_de_passe manquant")
    exit(1)

print("✅ Format DATABASE_URL valide")
```

### Méthode 3 : Test de connexion (si vous avez accès au serveur)

```bash
# Depuis le serveur de production
docker compose -f docker-compose.production.yml exec backend python -c "
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print('❌ DATABASE_URL non défini')
    exit(1)

print(f'DATABASE_URL: {DATABASE_URL[:50]}...')  # Afficher les 50 premiers caractères

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1'))
        print('✅ Connexion réussie à la base de données')
except Exception as e:
    print(f'❌ Erreur de connexion: {e}')
    exit(1)
"
```

## Vérification dans GitHub Actions

### Option 1 : DATABASE_URL n'est pas requis

**Bonne nouvelle** : `DATABASE_URL` n'est **pas obligatoire** dans votre configuration actuelle !

Le code Flask construit automatiquement `SQLALCHEMY_DATABASE_URI` depuis les variables `POSTGRES_*` si `DATABASE_URL` n'est pas défini. Voir `backend/config.py` ligne 227-229 :

```python
SQLALCHEMY_DATABASE_URI = (
    _db_url_from_secret if _db_url_from_secret else _build_database_url_safe()
)
```

Donc vous avez deux options :

1. **Ne pas définir DATABASE_URL** : Le code construira l'URL depuis `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
2. **Définir DATABASE_URL** : Pour un contrôle explicite

### Option 2 : Si vous voulez définir DATABASE_URL

Construisez-le depuis vos secrets existants :

```bash
# Dans votre workflow GitHub Actions ou localement
export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"
```

**Note** : Le workflow de déploiement construit déjà `DATABASE_URL` automatiquement depuis `SQLALCHEMY_DATABASE_URI` (ligne 301 du workflow).

## Vérification rapide

### Checklist de validation

- [ ] Format commence par `postgresql://`
- [ ] Contient `user:password@host:port/database`
- [ ] Le host est `postgres` (nom du service Docker) ou l'IP du serveur
- [ ] Le port est `5432` (port PostgreSQL standard)
- [ ] Le nom de la base correspond à `POSTGRES_DB`
- [ ] Le nom d'utilisateur correspond à `POSTGRES_USER`
- [ ] Le mot de passe correspond à `POSTGRES_PASSWORD` (avec échappement URL si caractères spéciaux)

## Caractères spéciaux dans le mot de passe

Si votre mot de passe PostgreSQL contient des caractères spéciaux, ils doivent être échappés :

| Caractère | Échappement URL |
| --------- | --------------- |
| `@`       | `%40`           |
| `#`       | `%23`           |
| `!`       | `%21`           |
| `%`       | `%25`           |
| `&`       | `%26`           |
| `+`       | `%2B`           |
| `=`       | `%3D`           |

**Exemple** :

- Mot de passe : `P@ssw0rd!#`
- Échappé : `P%40ssw0rd%21%23`
- DATABASE_URL : `postgresql://user:P%40ssw0rd%21%23@postgres:5432/db`

Le code Python fait cet échappement automatiquement via `urllib.parse.quote_plus()`.

## Recommandation

**Pour votre cas** : Vous n'avez **pas besoin** de définir `DATABASE_URL` dans GitHub Secrets si vous avez déjà :

- ✅ `POSTGRES_USER`
- ✅ `POSTGRES_PASSWORD`
- ✅ `POSTGRES_DB`

Le workflow construit automatiquement `DATABASE_URL` depuis ces variables (ligne 301 du workflow).

Cependant, si vous voulez un contrôle explicite, vous pouvez définir `DATABASE_URL` avec la valeur construite manuellement.
