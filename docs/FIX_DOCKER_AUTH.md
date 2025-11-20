# Résolution du problème d'authentification Docker

## 🔴 Problème

Erreur rencontrée :
```
docker: Error response from daemon: authentication required - incorrect username or password
```

Cette erreur se produit lors de la tentative de téléchargement de l'image `returntocorp/semgrep`, même si cette image est **publique** et ne nécessite normalement pas d'authentification.

## ✅ Solutions

### Solution 1 : Se déconnecter de Docker Hub (recommandée)

Si vous êtes connecté à Docker Hub avec des credentials incorrects, cela peut bloquer l'accès aux images publiques :

```powershell
# Se déconnecter de Docker Hub
docker logout

# Réessayer de télécharger l'image
docker pull returntocorp/semgrep:latest

# Puis exécuter Semgrep
cd C:\Users\jasiq\atmr\backend
docker run --rm -v "${PWD}:/src" -v "${PWD}/..:/project" -w /src returntocorp/semgrep semgrep --config=/project/.semgrep.yml --config=p/ci --config=p/security-audit .
```

### Solution 2 : Vérifier la configuration Docker Desktop

1. **Ouvrir Docker Desktop**
2. **Aller dans Settings → Docker Engine**
3. **Vérifier la configuration `registry-mirrors`**
   - Si elle pointe vers un registre privé, cela peut causer des problèmes
   - Vous pouvez temporairement la désactiver ou la supprimer
4. **Appliquer et redémarrer Docker Desktop**

### Solution 3 : Vérifier les credentials Docker

```powershell
# Vérifier si vous êtes connecté
docker info | Select-String "Username"

# Se déconnecter si nécessaire
docker logout

# Se reconnecter avec des credentials valides (si nécessaire pour des images privées)
docker login
```

### Solution 4 : Utiliser une alternative sans Docker

Si Docker pose problème, vous pouvez :

1. **Utiliser GitHub Actions** : Semgrep fonctionne déjà dans votre CI/CD
2. **Réparer Python** : Pour installer Semgrep localement (voir `docs/FIX_PYTHON_ISSUE.md`)
3. **Utiliser WSL** : Si Windows Subsystem for Linux est installé

### Solution 5 : Utiliser le script PowerShell corrigé

Un script a été créé qui gère automatiquement l'authentification :

```powershell
cd C:\Users\jasiq\atmr\backend
.\semgrep-simple.ps1
```

## 🔍 Vérification

Après avoir résolu le problème, vérifiez :

```powershell
# Vérifier que Docker fonctionne
docker info

# Télécharger l'image Semgrep
docker pull returntocorp/semgrep:latest

# Vérifier que l'image est bien téléchargée
docker images | Select-String "semgrep"
```

## 📋 Commandes rapides

### Déconnexion et réessai

```powershell
# Se déconnecter
docker logout

# Télécharger l'image
docker pull returntocorp/semgrep:latest

# Exécuter Semgrep
cd C:\Users\jasiq\atmr\backend
docker run --rm -v "${PWD}:/src" -v "${PWD}/..:/project" -w /src returntocorp/semgrep semgrep --config=/project/.semgrep.yml --config=p/ci --config=p/security-audit .
```

## 🎯 Note importante

**Rappel** : Semgrep fonctionne déjà dans votre CI/CD GitHub Actions ! 

Le problème d'authentification Docker n'affecte que votre environnement local. Vous pouvez :
- ✅ Continuer à développer sans Semgrep local
- ✅ Laisser GitHub Actions faire les scans automatiquement lors des PR
- ✅ Résoudre le problème Docker quand vous avez besoin de scanner localement

## 📚 Ressources

- [Documentation Docker authentication](https://docs.docker.com/engine/reference/commandline/login/)
- [Troubleshooting Docker Hub access](https://docs.docker.com/docker-hub/troubleshoot/)
- [Semgrep Docker image](https://hub.docker.com/r/returntocorp/semgrep)

