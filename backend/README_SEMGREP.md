# Utilisation de Semgrep

## 🚀 Méthode la plus simple : Commande directe

Depuis le répertoire `backend` :

```powershell
cd C:\Users\jasiq\atmr\backend

docker run --rm -v "C:\Users\jasiq\atmr\backend:/src:ro" -w /src returntocorp/semgrep:latest semgrep --config=p/ci --config=p/security-audit .
```

## 📋 Scripts PowerShell disponibles

### Script simplifié (recommandé)

```powershell
.\semgrep-simple.ps1
```

## ⏱️ Temps d'exécution

**⚠️ IMPORTANT : Semgrep peut prendre 2-5 minutes à s'exécuter** sur un projet de taille moyenne.

### Facteurs qui influencent le temps :

- **Nombre de fichiers** : Plus il y a de fichiers Python, plus c'est long
- **Complexité des règles** : Les règles de sécurité (`p/security-audit`) sont plus lentes
- **Taille des fichiers** : Les gros fichiers prennent plus de temps
- **Docker** : L'overhead Docker ajoute un peu de temps

### Optimisations :

Le fichier `.semgrepignore` exclut déjà :
- Les fichiers générés (`__pycache__`, `.pyc`)
- Les migrations de base de données
- Les fichiers de cache
- Les modèles ML volumineux

**Pour accélérer** (si nécessaire), vous pouvez exclure les tests :
```bash
# Décommenter dans .semgrepignore :
**/tests/**
**/test_*.py
```

## ⚙️ Résolution des problèmes PowerShell

Si vous voyez des erreurs comme "cannot be loaded because running scripts is disabled", modifiez la politique d'exécution :

```powershell
# Vérifier la politique actuelle
Get-ExecutionPolicy

# Pour la session actuelle uniquement (temporaire, recommandé)
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Puis exécuter le script
.\semgrep-simple.ps1
```

## 📊 Interprétation des résultats

- **Code 0** : ✅ Aucun problème détecté
- **Code 1** : ⚠️ Problèmes de sécurité trouvés (consultez les résultats ci-dessus)
- **Autre** : ❌ Erreur lors de l'exécution

## 🔍 Configuration Semgrep

Semgrep utilise :
- `p/ci` : Règles de qualité de code
- `p/security-audit` : Règles de sécurité OWASP

Les suppressions inline (`# nosemgrep`) dans le code sont reconnues par Semgrep.

## 💡 Conseils

1. **Patience** : Le premier scan peut prendre 2-5 minutes
2. **CI/CD** : Dans GitHub Actions, les scans sont exécutés automatiquement
3. **Local** : Vous n'avez pas besoin de scanner tout le temps localement
4. **Optimisation** : Le fichier `.semgrepignore` réduit déjà le temps d'exécution
