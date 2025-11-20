# Résolution du problème Python "ModuleNotFoundError: No module named 'encodings'"

## 🔴 Problème

Erreur rencontrée :

```
Could not find platform independent libraries <prefix>
Fatal Python error: Failed to import encodings module
ModuleNotFoundError: No module named 'encodings'
```

Cette erreur indique que l'installation de Python est **corrompue ou incomplète**.

## ✅ Solutions

### Solution 1 : Réinstaller Python (recommandée)

1. **Désinstaller Python complètement** :

   - Ouvrir "Paramètres" → "Applications" → "Applications et fonctionnalités"
   - Rechercher "Python" et désinstaller toutes les versions
   - Supprimer les dossiers Python restants :
     - `C:\Users\VotreNom\AppData\Local\Programs\Python`
     - `C:\Python*`

2. **Télécharger Python depuis le site officiel** :

   - Aller sur https://www.python.org/downloads/
   - Télécharger Python 3.11+ (version stable recommandée)

3. **Installer Python correctement** :

   - ⚠️ **IMPORTANT** : Cocher "Add Python to PATH" lors de l'installation
   - Cocher "Install launcher for all users" (optionnel mais recommandé)
   - Cliquer sur "Install Now"

4. **Vérifier l'installation** :

   ```powershell
   python --version
   pip --version
   ```

5. **Installer Semgrep** :
   ```powershell
   python -m pip install semgrep
   ```

### Solution 2 : Utiliser Docker (alternative rapide)

Si vous avez Docker installé, utilisez l'image Semgrep officielle :

```powershell
# Scanner le projet backend
docker run --rm -v "${PWD}\backend:/src" returntocorp/semgrep semgrep --config=auto /src

# OU avec la configuration du projet
docker run --rm -v "${PWD}:/src" returntocorp/semgrep semgrep --config=/src/.semgrep.yml --config=p/ci --config=p/security-audit /src/backend
```

**Avantages** :

- ✅ Pas besoin de corriger l'installation Python
- ✅ Semgrep préinstallé et configuré
- ✅ Isolation complète

**Inconvénient** :

- ❌ Nécessite Docker installé

### Solution 3 : Utiliser WSL (Windows Subsystem for Linux)

Si vous avez WSL installé :

```bash
# Dans WSL
sudo apt update
sudo apt install python3-pip
pip3 install semgrep

# Scanner le projet
cd /mnt/c/Users/jasiq/atmr/backend
semgrep --config=../.semgrep.yml --config=p/ci --config=p/security-audit .
```

### Solution 4 : Utiliser pipx (si pip fonctionne encore)

Essayez d'installer pipx d'abord :

```powershell
# Essayer d'installer pipx (peut échouer si Python est corrompu)
py -m pip install pipx

# Si pipx s'installe, utiliser pour installer Semgrep
pipx install semgrep
```

### Solution 5 : Réparer l'installation Python actuelle

1. **Ouvrir "Programmes et fonctionnalités"**
2. **Trouver Python dans la liste**
3. **Sélectionner "Modifier"** → **"Réparer"**

**Note** : Cette solution ne fonctionne pas toujours.

## 🎯 Solution recommandée pour votre cas

Étant donné que vous avez une installation Python corrompue, je recommande :

### Option A : Utiliser Docker (le plus rapide)

Si Docker est installé, c'est la solution la plus rapide :

```powershell
cd C:\Users\jasiq\atmr

# Scanner avec Semgrep via Docker
docker run --rm -v "${PWD}:/src" -w /src/backend returntocorp/semgrep semgrep --config=/src/.semgrep.yml --config=p/ci --config=p/security-audit .
```

### Option B : Réinstaller Python proprement (le plus fiable)

1. Désinstaller Python complètement
2. Télécharger et réinstaller depuis python.org
3. **Cocher "Add Python to PATH"**
4. Installer Semgrep : `python -m pip install semgrep`

## 📋 Vérification après correction

Une fois Python corrigé ou Docker utilisé, vérifiez :

```powershell
# Avec Python
python -m semgrep --version

# OU avec Docker
docker run --rm returntocorp/semgrep semgrep --version
```

## 🔧 Note sur CI/CD

**Bon point** : Semgrep fonctionne déjà dans votre CI/CD GitHub Actions !

Le problème n'affecte que votre environnement local Windows. Vous pouvez :

- ✅ Continuer à développer sans Semgrep local
- ✅ Laisser GitHub Actions faire les scans automatiquement
- ✅ Installer Semgrep plus tard quand Python sera corrigé

## 📚 Ressources

- [Documentation Python Windows](https://docs.python.org/3/using/windows.html)
- [Semgrep Docker](https://semgrep.dev/docs/getting-started/installation/#docker)
- [Troubleshooting Python encodings](https://bugs.python.org/issue29714)
