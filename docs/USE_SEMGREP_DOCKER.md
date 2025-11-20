# Utiliser Semgrep via Docker (Windows)

## 🐳 Prérequis

1. **Docker Desktop installé** : https://www.docker.com/products/docker-desktop/
2. **Docker Desktop démarré** : L'icône Docker doit être visible dans la barre des tâches

## 🚀 Utilisation

### Option 1 : Utiliser le script PowerShell (recommandé)

Un script PowerShell a été créé pour simplifier l'utilisation :

```powershell
# Aller dans le répertoire backend
cd C:\Users\jasiq\atmr\backend

# Exécuter Semgrep avec affichage normal
.\semgrep.ps1

# OU générer un rapport JSON
.\semgrep.ps1 -Json
```

### Option 2 : Utiliser Docker directement

```powershell
# Aller dans le répertoire backend
cd C:\Users\jasiq\atmr\backend

# Scanner avec affichage normal
docker run --rm -v "${PWD}:/src" -v "${PWD}/..:/project" -w /src returntocorp/semgrep semgrep --config=/project/.semgrep.yml --config=p/ci --config=p/security-audit .

# Scanner et générer un rapport JSON
docker run --rm -v "${PWD}:/src" -v "${PWD}/..:/project" -w /src returntocorp/semgrep semgrep --config=/project/.semgrep.yml --config=p/ci --config=p/security-audit . --json -o semgrep.json
```

## 📋 Explication des commandes Docker

### Syntaxe de base

```powershell
docker run --rm `
    -v "${PWD}:/src" `                    # Monter le répertoire backend dans /src
    -v "${PWD}/..:/project" `             # Monter le répertoire racine pour accéder à .semgrep.yml
    -w /src `                             # Définir /src comme répertoire de travail
    returntocorp/semgrep `                # Image Docker Semgrep officielle
    semgrep [options] .                   # Commande Semgrep
```

### Options principales

- `--rm` : Supprimer le conteneur après exécution
- `-v "${PWD}:/src"` : Monter le répertoire courant dans `/src` du conteneur
- `-v "${PWD}/..:/project"` : Monter le répertoire parent pour accéder à la config
- `-w /src` : Définir `/src` comme répertoire de travail
- `returntocorp/semgrep` : Image Docker Semgrep officielle

## 🔍 Options de Semgrep

### Scanner avec configuration personnalisée

```powershell
.\semgrep.ps1
```

### Générer un rapport JSON

```powershell
.\semgrep.ps1 -Json
```

### Scanner avec affichage verbose

```powershell
docker run --rm -v "${PWD}:/src" -v "${PWD}/..:/project" -w /src returntocorp/semgrep semgrep --config=/project/.semgrep.yml --config=p/ci --config=p/security-audit . --verbose
```

### Scanner uniquement les erreurs critiques

```powershell
docker run --rm -v "${PWD}:/src" -v "${PWD}/..:/project" -w /src returntocorp/semgrep semgrep --config=/project/.semgrep.yml --config=p/ci --config=p/security-audit . --severity ERROR
```

## ⚠️ Dépannage

### Erreur : "Docker n'est pas démarré"

**Solution** : Démarrer Docker Desktop et attendre qu'il soit prêt.

### Erreur : "Le fichier spécifié est introuvable"

**Solution** : Vérifier que Docker Desktop est bien démarré :
```powershell
docker info
```

### Erreur : "Cannot connect to Docker daemon"

**Solution** :
1. Vérifier que Docker Desktop est démarré
2. Redémarrer Docker Desktop
3. Vérifier les permissions utilisateur

### Erreur : "Path not found"

**Solution** : Utiliser des chemins absolus dans PowerShell :
```powershell
cd C:\Users\jasiq\atmr\backend
docker run --rm -v "C:\Users\jasiq\atmr\backend:/src" -v "C:\Users\jasiq\atmr:/project" -w /src returntocorp/semgrep semgrep --config=/project/.semgrep.yml --config=p/ci --config=p/security-audit .
```

## ✅ Avantages de Docker

- ✅ **Pas besoin de corriger Python** : Docker utilise son propre environnement
- ✅ **Isolation** : N'affecte pas votre système
- ✅ **Toujours à jour** : L'image Docker est maintenue par Semgrep
- ✅ **Multi-plateforme** : Fonctionne de la même manière sur Windows, Mac, Linux

## 📚 Ressources

- [Documentation Semgrep Docker](https://semgrep.dev/docs/getting-started/installation/#docker)
- [Docker Desktop pour Windows](https://docs.docker.com/desktop/install/windows-install/)

