# Workflows Docker - Production vs RL

Ce document explique comment les images Docker sont construites et déployées.

## 📦 Images Docker

### Image Production (`djasiqi/atmr-backend:latest`)

- **Dockerfile** : `backend/Dockerfile.production`
- **Requirements** : `requirements.prod.txt` (inclut `requirements.base.txt`)
- **Taille** : ~500MB (sans torch/CUDA)
- **Contenu** : Flask, SQLAlchemy, Celery, Redis, scikit-learn, etc.
- **Pas de** : torch, gymnasium, optuna, pytest

### Image RL (`djasiqi/atmr-backend:rl-latest`)

- **Dockerfile** : `backend/Dockerfile.rl`
- **Requirements** : `requirements-rl.txt` (inclut `requirements.base.txt` + torch, gymnasium, optuna)
- **Taille** : ~2GB (avec PyTorch CPU-only)
- **Contenu** : Tout de la prod + torch, gymnasium, optuna, tensorboard, matplotlib
- **Usage** : Training RL, expériences ML

## 🔄 Workflows GitHub Actions

### 1. Workflow Production (`.github/workflows/deploy.yml`)

**Déclenchement :**

- `workflow_dispatch` (manuel)
- ~~`push` sur `main`~~ (actuellement désactivé)

**Job : `build-and-push`**

- **Runner** : `ubuntu-latest` (GitHub Actions standard)
- **Build** : `Dockerfile.production`
- **Build args** :
  ```yaml
  WITH_RL=false
  RL_ENABLED=false
  REQUIREMENTS_HASH=<hash>
  ```
- **Tags** : `${{ secrets.DOCKER_IMAGE }}:${{ secrets.DOCKER_TAG }}` (ex: `djasiqi/atmr-backend:latest`)
- **Push** : Oui, vers Docker Hub
- **Déploiement** : Automatique sur le serveur de production via SSH

**Étapes principales :**

1. Checkout code
2. Nettoyage espace disque
3. Build image production (légère)
4. Push vers Docker Hub
5. Déploiement sur serveur production

### 2. Workflow RL (`.github/workflows/build-rl.yml`)

**Déclenchement :**

- `workflow_dispatch` (manuel uniquement)
- **Inputs** :
  - `tag` : Tag de l'image (défaut: `rl-latest`)
  - `push` : Push vers Docker Hub (défaut: `false`)

**Job : `build-rl-image`**

- **Runner** : `ubuntu-latest` (ou `self-hosted` si configuré)
- **Build** : `Dockerfile.rl`
- **Build args** :
  ```yaml
  WITH_RL=true
  RL_ENABLED=true
  REQUIREMENTS_HASH=<hash>
  ```
- **Tags** : `${{ secrets.DOCKER_IMAGE }}:${{ inputs.tag }}` (ex: `djasiqi/atmr-backend:rl-latest`)
- **Push** : Optionnel (selon input `push`)

**⚠️ Important :**

- Cette image est **lourde** (~2GB avec PyTorch)
- Risque de "No space left on device" sur les runners GitHub standards
- **Recommandation** : Utiliser un self-hosted runner ou build local

**Étapes principales :**

1. Checkout code
2. Nettoyage espace disque (critique pour RL)
3. Build image RL (lourde)
4. Push vers Docker Hub (optionnel)
5. Vérification taille image

## 🚀 Utilisation

### Build Production (automatique)

```bash
# Via GitHub Actions UI
# Actions > Build & Deploy > Run workflow
```

Ou déclenché automatiquement sur push (quand activé).

### Build RL (manuel)

```bash
# Via GitHub Actions UI
# Actions > Build & Push RL Image > Run workflow
# Options:
#   - tag: rl-latest (ou autre)
#   - push: true (pour pousser vers Docker Hub)
```

### Build RL Local (recommandé)

Si vous avez un serveur avec plus de ressources :

```bash
# Sur votre serveur 8vCPU/16GB
cd backend
docker build -f Dockerfile.rl -t djasiqi/atmr-backend:rl-latest .
docker push djasiqi/atmr-backend:rl-latest
```

## 📊 Comparaison

| Aspect            | Production              | RL                               |
| ----------------- | ----------------------- | -------------------------------- |
| **Workflow**      | `deploy.yml`            | `build-rl.yml`                   |
| **Dockerfile**    | `Dockerfile.production` | `Dockerfile.rl`                  |
| **Requirements**  | `requirements.prod.txt` | `requirements-rl.txt`            |
| **Taille**        | ~500MB                  | ~2GB                             |
| **Déclenchement** | Auto (push) ou manuel   | Manuel uniquement                |
| **Push**          | Oui (automatique)       | Optionnel                        |
| **Déploiement**   | Automatique (SSH)       | Manuel                           |
| **Runner**        | `ubuntu-latest`         | `ubuntu-latest` ou `self-hosted` |

## 🔧 Configuration Self-Hosted Runner (optionnel)

Pour éviter les problèmes d'espace disque avec l'image RL :

1. **Créer un self-hosted runner** sur votre serveur
2. **Modifier** `.github/workflows/build-rl.yml` :
   ```yaml
   runs-on: self-hosted # Au lieu de ubuntu-latest
   ```

Ou utiliser l'input `use_self_hosted` pour choisir dynamiquement.

## ✅ Vérification

### Production

```bash
# Vérifier que l'image prod n'a pas torch
docker run --rm djasiqi/atmr-backend:latest python -c "import torch" 2>&1
# Devrait échouer: ModuleNotFoundError: No module named 'torch'
```

### RL

```bash
# Vérifier que l'image RL a torch
docker run --rm djasiqi/atmr-backend:rl-latest python -c "import torch; print(f'PyTorch {torch.__version__}')"
# Devrait afficher: PyTorch 2.x.x
```

## 🎯 Résumé

- **Production** : Build automatique, léger, déploiement automatique
- **RL** : Build manuel, lourd, déploiement manuel (ou sur self-hosted runner)

La séparation est claire : production = rapide et automatique, RL = lourd et manuel.
