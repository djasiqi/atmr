# Installation de Semgrep

## 📋 Méthodes d'installation

### Option 1 : Installation via pip (recommandée)

**Prérequis** : Python 3.7+ installé

```bash
# Installer Semgrep globalement
pip install semgrep

# OU installer pour l'utilisateur uniquement
pip install --user semgrep
```

**Vérifier l'installation** :

```bash
semgrep --version
```

### Option 2 : Installation via pipx (isolée, recommandée)

**Prérequis** : pipx installé

```bash
# Installer pipx si nécessaire
python -m pip install --user pipx
python -m pipx ensurepath

# Installer Semgrep via pipx
pipx install semgrep
```

**Avantages** : Installation isolée, ne pollue pas l'environnement Python global

### Option 3 : Installation via Homebrew (macOS/Linux)

```bash
brew install semgrep
```

### Option 4 : Installation via Scoop (Windows)

```bash
scoop install semgrep
```

### Option 5 : Installation via Chocolatey (Windows)

```bash
choco install semgrep
```

### Option 6 : Installation via Docker

```bash
docker run --rm -v "${PWD}:/src" returntocorp/semgrep semgrep --config=auto /src
```

## 🔧 Installation sur Windows

### Si Python n'est pas installé :

1. **Télécharger Python** :

   - Aller sur https://www.python.org/downloads/
   - Télécharger Python 3.11+ pour Windows
   - ⚠️ **Important** : Cocher "Add Python to PATH" lors de l'installation

2. **Installer Semgrep** :

   ```powershell
   python -m pip install semgrep
   ```

3. **Vérifier l'installation** :
   ```powershell
   python -m pip show semgrep
   semgrep --version
   ```

### Si Python est installé mais pas dans le PATH :

1. **Utiliser py launcher** :

   ```powershell
   py -m pip install semgrep
   ```

2. **Utiliser le chemin complet** :
   ```powershell
   C:\Python311\python.exe -m pip install semgrep
   ```

### Alternative : Installation via pipx (recommandée pour Windows)

1. **Installer pipx** :

   ```powershell
   py -m pip install --user pipx
   py -m pipx ensurepath
   ```

2. **Installer Semgrep** :

   ```powershell
   pipx install semgrep
   ```

3. **Fermer et rouvrir le terminal** pour que le PATH soit mis à jour

## ✅ Vérification de l'installation

Une fois installé, vérifiez que Semgrep fonctionne :

```bash
# Vérifier la version
semgrep --version

# Tester avec un scan simple
semgrep --version

# Scanner un répertoire
cd backend
semgrep --config p/ci .
```

## 🚀 Utilisation avec le projet ATMR

Une fois Semgrep installé, vous pouvez l'utiliser avec la configuration du projet :

```bash
# Depuis le répertoire backend
cd backend

# Scanner avec la configuration du projet
semgrep --config ../.semgrep.yml --config p/ci --config p/security-audit .

# Générer un rapport JSON
semgrep --config ../.semgrep.yml --config p/ci --config p/security-audit . --json -o semgrep.json

# Afficher uniquement les findings critiques
semgrep --config ../.semgrep.yml --config p/ci --config p/security-audit . --severity ERROR
```

## 🔍 Dépannage

### Erreur : "semgrep n'est pas reconnu"

**Causes possibles** :

1. Semgrep n'est pas installé
2. Le répertoire Scripts de Python n'est pas dans le PATH
3. Le terminal n'a pas été redémarré après l'installation

**Solutions** :

1. **Vérifier si Semgrep est installé** :

   ```powershell
   py -m pip show semgrep
   ```

2. **Réinstaller en forçant** :

   ```powershell
   py -m pip install --upgrade --force-reinstall semgrep
   ```

3. **Ajouter Python au PATH manuellement** :

   - Ouvrir "Variables d'environnement" dans Windows
   - Ajouter `C:\Users\VotreNom\AppData\Local\Programs\Python\Python311\Scripts` au PATH
   - Redémarrer le terminal

4. **Utiliser py -m semgrep** :
   ```powershell
   py -m semgrep --version
   ```

### Erreur : "pip n'est pas reconnu"

**Solution** :

```powershell
# Utiliser python -m pip au lieu de pip directement
python -m pip install semgrep
# OU
py -m pip install semgrep
```

### Erreur : "Permission denied" (Linux/macOS)

**Solution** :

```bash
# Utiliser --user pour installer pour l'utilisateur uniquement
pip install --user semgrep

# OU utiliser sudo (non recommandé)
sudo pip install semgrep
```

## 📚 Ressources

- [Documentation officielle Semgrep](https://semgrep.dev/docs/getting-started/)
- [Guide d'installation complet](https://semgrep.dev/docs/getting-started/installation/)
- [Troubleshooting](https://semgrep.dev/docs/getting-started/troubleshooting/)
