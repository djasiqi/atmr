# Rapport d'Analyse et Correction - Pipeline CI/CD Build & Push

**Date d'analyse** : 2025-11-21  
**Workflow** : build-and-push  
**Run ID** : 19576500860  
**Statut global** : ❌ **ÉCHEC** (migrations échouées)

---

## 📋 Résumé Exécutif

Le pipeline CI/CD présente **1 erreur critique** bloquant le déploiement, **3 warnings majeurs** et plusieurs anomalies mineures. L'échec principal survient lors de l'exécution des migrations Alembic : le mot de passe PostgreSQL contient des caractères spéciaux (`37_46!!`) qui ne sont pas correctement échappés dans la chaîne de connexion `DATABASE_URL`, provoquant une erreur de résolution DNS (`could not translate host name "37_46!!@postgres"`).

**Impact** : Le déploiement échoue systématiquement, empêchant toute mise à jour en production.

**Priorité des corrections** :

1. 🔴 **CRITIQUE** : Échappement URL du mot de passe PostgreSQL dans DATABASE_URL
2. 🟠 **MAJEUR** : Mise à jour Trivy (0.65.0 → 0.67.2)
3. 🟠 **MAJEUR** : Gestion des conteneurs orphelins (nginx)
4. 🟡 **MINEUR** : Warnings pip root user, useradd UID

---

## 📊 Tableau des Anomalies Détectées

| Step                             | Type    | Gravité         | Log brut                                                                                       | Cause probable                                        | Fix rapide                                                       |
| -------------------------------- | ------- | --------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------- |
| **12_Deploy via SSH**            | ERROR   | 🔴 **CRITIQUE** | `psycopg2.OperationalError: could not translate host name "37_46!!@postgres"`                  | Mot de passe PostgreSQL non échappé dans DATABASE_URL | Échapper POSTGRES_PASSWORD avec `urllib.parse.quote()`           |
| **9_Scan image with Trivy**      | WARNING | 🟠 **MAJEUR**   | `Version 0.67.2 of Trivy is now available, current version is 0.65.0`                          | Version Trivy obsolète                                | Mettre à jour `version: v0.67.2` dans workflow                   |
| **12_Deploy via SSH**            | WARNING | 🟠 **MAJEUR**   | `Found orphan containers ([***-nginx]) for this project`                                       | Conteneur nginx orphelin (commenté dans compose)      | Ajouter `--remove-orphans` ou supprimer le conteneur             |
| **12_Deploy via SSH**            | ERROR   | 🟠 **MAJEUR**   | `ModuleNotFoundError: No module named 'gymnasium'` (10 erreurs tests)                          | Tests RL exécutés alors que WITH_RL=false             | Exclure tests RL du stage testing ou conditionner leur exécution |
| **23_Post Set up QEMU**          | WARNING | 🟡 **MINEUR**   | `Failed to save: Unable to reserve cache with key docker.io--tonistiigi--binfmt-***-linux-x64` | Conflit de cache concurrent                           | Non bloquant, cache sera restauré au prochain run                |
| **8_Build & push backend image** | WARNING | 🟡 **MINEUR**   | `WARNING: Running pip as the 'root' user can result in broken permissions`                     | Pip exécuté en root dans stage builder                | Ajouter `--root-user-action=ignore` ou utiliser venv             |
| **8_Build & push backend image** | WARNING | 🟡 **MINEUR**   | `useradd warning: appuser's uid 10001 is greater than SYS_UID_MAX 999`                         | UID système > 999                                     | Utiliser UID < 1000 (ex: 999)                                    |
| **4_Checkout**                   | INFO    | 🟢 **INFO**     | `hint: Using 'master' as the name for the initial branch`                                      | Git utilise master par défaut                         | Non bloquant, suggestion de config                               |

---

## 🔍 Analyse par Root Cause

### Root Cause #1 : Échappement URL manquant pour POSTGRES_PASSWORD dans DATABASE_URL

**Symptômes observés** :

```
psycopg2.OperationalError: could not translate host name "37_46!!@postgres" to address: Name or service not known
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not translate host name "37_46!!@postgres" to address
```

**Mécanisme exact** :

- Le fichier `docker-compose.production.yml` ligne 78 construit `DATABASE_URL` ainsi :
  ```yaml
  DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
  ```
- Si `POSTGRES_PASSWORD` contient des caractères spéciaux (`37_46!!`), ils sont injectés directement sans encodage URL.
- psycopg2 interprète `37_46!!@postgres` comme un hostname au lieu de `postgres` avec un mot de passe échappé.

**Conséquences si non corrigé** :

- ❌ Déploiements impossibles en production
- ❌ Migrations Alembic échouent systématiquement
- ❌ Application non fonctionnelle après déploiement

**Fichiers concernés** :

- `docker-compose.production.yml` (ligne 78, 150, 193)
- Script de déploiement SSH (workflow GitHub Actions)

---

### Root Cause #2 : Version Trivy obsolète

**Symptômes observés** :

```
📣 Notices:
  - Version 0.67.2 of Trivy is now available, current version is 0.65.0
```

**Mécanisme exact** :

- Le workflow utilise `version: v0.65.0` alors que la version 0.67.2 est disponible.
- Risque de faux négatifs sur des vulnérabilités récentes.

**Conséquences si non corrigé** :

- ⚠️ Détection de vulnérabilités incomplète
- ⚠️ Exposition à des failles de sécurité non détectées

**Fichiers concernés** :

- `.github/workflows/deploy.yml` (ou workflow équivalent)

---

### Root Cause #3 : Conteneur nginx orphelin

**Symptômes observés** :

```
Found orphan containers ([***-nginx]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
```

**Mécanisme exact** :

- Le service `nginx` est commenté dans `docker-compose.production.yml` (lignes 238-260).
- Un conteneur nginx existe encore sur le serveur de production.
- Docker Compose détecte l'orphelin mais ne le supprime pas automatiquement.

**Conséquences si non corrigé** :

- ⚠️ Confusion lors des déploiements
- ⚠️ Consommation de ressources inutile
- ⚠️ Potentiels conflits de ports

**Fichiers concernés** :

- Script de déploiement SSH (ajout de `--remove-orphans`)
- `docker-compose.production.yml` (décommenter ou supprimer définitivement)

---

### Root Cause #4 : Tests RL exécutés alors que WITH_RL=false

**Symptômes observés** :

```
ModuleNotFoundError: No module named 'gymnasium'
ERROR tests/rl/test_dispatch_env*.py (10 erreurs)
```

**Mécanisme exact** :

- Le build Docker utilise `WITH_RL=false` (ligne 5 du log "8_Build & push backend image.txt").
- Le stage `testing` du Dockerfile n'exclut pas les tests RL.
- Lors de l'exécution des tests (probablement via pytest discovery), les imports de `gymnasium` échouent.

**Conséquences si non corrigé** :

- ⚠️ Logs pollués par des erreurs de tests non pertinents
- ⚠️ Confusion lors du debugging

**Fichiers concernés** :

- `backend/Dockerfile.production` (stage testing)
- `backend/pytest.ini` (exclure tests RL si WITH_RL=false)

---

## 🛠️ Plan de Correction Étape-par-Étape

### Étape 1 : Corriger l'échappement URL de POSTGRES_PASSWORD (CRITIQUE)

**Fichier** : `docker-compose.production.yml`

**Changement** : Utiliser une fonction shell pour échapper le mot de passe, ou mieux, construire DATABASE_URL côté application.

**Option A (Recommandée)** : Ne pas construire DATABASE_URL dans docker-compose, laisser l'application le faire.

```yaml
# AVANT (ligne 78)
DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
# APRÈS
# Ne pas définir DATABASE_URL ici, laisser l'application le construire depuis les variables individuelles
# OU utiliser un script d'échappement
```

**Option B** : Script d'échappement dans le workflow SSH.

**Fichier** : Workflow GitHub Actions (script SSH)

```bash
# AVANT
export POSTGRES_PASSWORD="***"
DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"

# APRÈS
export POSTGRES_PASSWORD="***"
# Échapper le mot de passe pour URL
POSTGRES_PASSWORD_ESCAPED=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${POSTGRES_PASSWORD}', safe=''))")
DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD_ESCAPED}@postgres:5432/${POSTGRES_DB}"
export DATABASE_URL
```

**Option C (Meilleure)** : Modifier `backend/config.py` pour construire DATABASE_URL depuis les variables individuelles si DATABASE_URL n'est pas définie.

**Fichier** : `backend/config.py`

```python
# Ajouter après ligne 126
def _build_database_url():
    """Construit DATABASE_URL depuis les variables individuelles avec échappement URL."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    # Construire depuis variables individuelles
    user = os.getenv("POSTGRES_USER", "atmr_user")
    password = os.getenv("POSTGRES_PASSWORD", "atmr_password")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "atmr_db")

    # Échapper le mot de passe
    from urllib.parse import quote_plus
    password_escaped = quote_plus(password)

    return f"postgresql://{user}:{password_escaped}@{host}:{port}/{db}"

SQLALCHEMY_DATABASE_URI = _build_database_url()
```

**Validation** :

```bash
# Tester avec un mot de passe contenant des caractères spéciaux
export POSTGRES_PASSWORD="37_46!!"
export POSTGRES_USER="atmr_user"
export POSTGRES_DB="atmr_db"
python3 -c "from urllib.parse import quote_plus; print(f'postgresql://atmr_user:{quote_plus(\"37_46!!\")}@postgres:5432/atmr_db')"
# Doit afficher: postgresql://atmr_user:37_46%21%21@postgres:5432/atmr_db
```

---

### Étape 2 : Mettre à jour Trivy (MAJEUR)

**Fichier** : `.github/workflows/deploy.yml` (ou workflow équivalent)

**Changement** :

```yaml
# AVANT
- uses: aquasecurity/trivy-action@master
  with:
    version: v0.65.0

# APRÈS
- uses: aquasecurity/trivy-action@master
  with:
    version: v0.67.2
```

**Validation** :

- Vérifier que le scan Trivy s'exécute sans erreur
- Vérifier l'absence du warning de version

---

### Étape 3 : Gérer les conteneurs orphelins (MAJEUR)

**Fichier** : Script de déploiement SSH (workflow GitHub Actions)

**Changement** :

```bash
# AVANT
docker compose -f docker-compose.production.yml down || true

# APRÈS
docker compose -f docker-compose.production.yml down --remove-orphans || true
```

**Alternative** : Supprimer manuellement le conteneur nginx sur le serveur.

```bash
docker stop ***-nginx 2>/dev/null || true
docker rm ***-nginx 2>/dev/null || true
```

**Validation** :

- Vérifier l'absence du warning "orphan containers" dans les logs

---

### Étape 4 : Exclure les tests RL si WITH_RL=false (MAJEUR)

**Fichier** : `backend/Dockerfile.production` (stage testing)

**Changement** :

```dockerfile
# AVANT (ligne 277)
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

# APRÈS
CMD ["sh", "-c", "if [ \"$WITH_RL\" = \"false\" ]; then pytest tests/ -v --tb=short --ignore=tests/rl --ignore=tests/e2e/test_dispatch_e2e.py --ignore=tests/e2e/test_dispatch_metrics_e2e.py; else pytest tests/ -v --tb=short; fi"]
```

**Alternative** : Modifier `backend/pytest.ini`

```ini
# Ajouter
[pytest]
# Exclure tests RL si WITH_RL=false
markers =
    rl: tests nécessitant gymnasium (nécessite WITH_RL=true)

# Dans conftest.py, ajouter:
import pytest
import os

def pytest_configure(config):
    if os.getenv("WITH_RL", "true").lower() == "false":
        config.option.markexpr = "not rl"
```

**Validation** :

- Vérifier que les tests RL ne s'exécutent pas lorsque WITH_RL=false
- Vérifier l'absence d'erreurs `ModuleNotFoundError: No module named 'gymnasium'`

---

### Étape 5 : Corriger les warnings mineurs (MINEUR)

#### 5.1 : Warning pip root user

**Fichier** : `backend/Dockerfile.production` (stage builder)

**Changement** :

```dockerfile
# AVANT (ligne 47)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# APRÈS
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels --root-user-action=ignore -r requirements.txt
```

#### 5.2 : Warning useradd UID

**Fichier** : `backend/Dockerfile.production` (ligne 173)

**Changement** :

```dockerfile
# AVANT
useradd -r -g appgroup -u 10001 -d /app -s /bin/bash -c "ATMR App User" appuser

# APRÈS
useradd -r -g appgroup -u 999 -d /app -s /bin/bash -c "ATMR App User" appuser
```

**Validation** :

- Vérifier l'absence des warnings dans les logs de build

---

## 📝 Patchs / Snippets Consolidés

### Patch 1 : docker-compose.production.yml (Échappement DATABASE_URL)

```yaml
# Remplacer toutes les occurrences de DATABASE_URL construites manuellement
# Par des variables individuelles et laisser l'application construire l'URL

# backend service (ligne ~78)
environment:
  # Supprimer DATABASE_URL, utiliser variables individuelles
  POSTGRES_USER: ${POSTGRES_USER:-atmr_user}
  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-atmr_password}
  POSTGRES_HOST: postgres
  POSTGRES_PORT: 5432
  POSTGRES_DB: ${POSTGRES_DB:-atmr_db}
  # DATABASE_URL sera construit par config.py avec échappement
# Idem pour celery-worker (ligne ~150) et celery-beat (ligne ~193)
```

### Patch 2 : backend/config.py (Construction DATABASE_URL avec échappement)

```python
# Ajouter cette fonction avant la classe Config
from urllib.parse import quote_plus

def _build_database_url_safe():
    """Construit DATABASE_URL depuis variables individuelles avec échappement URL."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    user = os.getenv("POSTGRES_USER", "atmr_user")
    password = os.getenv("POSTGRES_PASSWORD", "atmr_password")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "atmr_db")

    password_escaped = quote_plus(password)
    return f"postgresql://{user}:{password_escaped}@{host}:{port}/{db}"

# Dans ProductionConfig (ligne ~126)
SQLALCHEMY_DATABASE_URI = _get_secret_from_vault_or_env(
    vault_path="prod/database/url",
    vault_key="value",
    env_key="DATABASE_URL",
    default=_build_database_url_safe()  # Utiliser la fonction au lieu d'une string
)
```

### Patch 3 : Workflow GitHub Actions (Script SSH - Ajout --remove-orphans)

```yaml
# Dans le step "Deploy via SSH"
script: |
  cd /srv/***
  set -o errexit -o nounset -o pipefail -x
  # ... exports ...
  docker compose -f docker-compose.production.yml down --remove-orphans || true
  docker compose -f docker-compose.production.yml up -d
  # ... reste du script ...
```

### Patch 4 : Workflow GitHub Actions (Mise à jour Trivy)

```yaml
- uses: aquasecurity/trivy-action@master
  with:
    version: v0.67.2 # Au lieu de v0.65.0
```

### Patch 5 : backend/Dockerfile.production (Exclure tests RL)

```dockerfile
# Stage testing (ligne ~277)
CMD ["sh", "-c", "if [ \"$WITH_RL\" = \"false\" ]; then pytest tests/ -v --tb=short --ignore=tests/rl --ignore-glob='tests/e2e/test_dispatch*e2e.py'; else pytest tests/ -v --tb=short; fi"]
```

---

## ✅ Plan de Validation & Non-Régression

### Tests de validation

1. **Test échappement mot de passe** :

   ```bash
   export POSTGRES_PASSWORD="37_46!!@test"
   python3 -c "from urllib.parse import quote_plus; print(quote_plus('$POSTGRES_PASSWORD'))"
   # Doit afficher: 37_46%21%21%40test
   ```

2. **Test connexion PostgreSQL** :

   ```bash
   # Avec mot de passe échappé
   DATABASE_URL="postgresql://user:37_46%21%21@postgres:5432/db" python3 -c "from sqlalchemy import create_engine; engine = create_engine('$DATABASE_URL'); engine.connect()"
   ```

3. **Test déploiement complet** :

   - Déclencher le workflow GitHub Actions
   - Vérifier que les migrations s'exécutent sans erreur
   - Vérifier l'absence du warning "orphan containers"
   - Vérifier l'absence du warning Trivy version

4. **Test exclusion tests RL** :
   ```bash
   docker build --build-arg WITH_RL=false -t test-image .
   docker run --rm test-image pytest tests/ --collect-only | grep -i rl
   # Ne doit pas lister de tests RL
   ```

### Garde-fous à ajouter

1. **set -euo pipefail** : Déjà présent dans le script SSH ✅

2. **Healthcheck Postgres** : Déjà présent dans docker-compose ✅

3. **Vérification tag Docker** :

   ```bash
   # Ajouter dans le workflow après le build
   - name: Verify Docker image tag
     run: |
       docker pull ${DOCKER_IMAGE}:${DOCKER_TAG}
       docker inspect ${DOCKER_IMAGE}:${DOCKER_TAG} | jq -r '.[0].RepoDigests[0]'
   ```

4. **Politique Trivy (fail-on-severity)** :

   ```yaml
   - uses: aquasecurity/trivy-action@master
     with:
       version: v0.67.2
       severity: CRITICAL,HIGH # Échouer sur CRITICAL et HIGH
       exit-code: 1
   ```

5. **Validation migrations Alembic** :
   ```bash
   # Ajouter dans le script SSH après les migrations
   docker compose -f docker-compose.production.yml exec -T backend flask db current
   docker compose -f docker-compose.production.yml exec -T backend flask db heads
   # Vérifier que current == heads
   ```

---

## 🎯 Conclusion

**État réel du pipeline** : Le pipeline échoue systématiquement à l'étape des migrations en raison d'un problème d'échappement URL du mot de passe PostgreSQL. Les autres anomalies sont non-bloquantes mais doivent être corrigées pour améliorer la robustesse et la sécurité.

**Niveau de risque** : 🔴 **ÉLEVÉ** - Déploiements impossibles en production.

**Priorité des fixes** :

1. 🔴 **IMMÉDIATE** : Échappement URL POSTGRES_PASSWORD (bloquant)
2. 🟠 **URGENTE** : Mise à jour Trivy, gestion conteneurs orphelins (sécurité/robustesse)
3. 🟡 **IMPORTANTE** : Exclusion tests RL, corrections warnings mineurs (qualité)

**Temps estimé de correction** : 2-4 heures (dont tests et validation).

---

**Rapport généré le** : 2025-11-21  
**Analysé par** : Staff Engineer DevOps/Backend
