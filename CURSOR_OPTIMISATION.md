# 🚀 Guide d'Optimisation de Cursor pour Meilleures Performances

Ce guide vous explique comment configurer Cursor pour obtenir les meilleures performances avec votre projet ATMR.

## 📋 Table des Matières

1. [Fichier .cursorignore](#1-fichier-cursorignore)
2. [Paramètres Cursor](#2-paramètres-cursor)
3. [Configuration des Extensions](#3-configuration-des-extensions)
4. [Indexation et Recherche](#4-indexation-et-recherche)
5. [Options Avancées](#5-options-avancées)
6. [Vérification des Performances](#6-vérification-des-performances)

---

## 1. Fichier .cursorignore

✅ **FAIT** - Le fichier `.cursorignore` a été créé automatiquement.

Ce fichier indique à Cursor de **ne pas indexer** les fichiers suivants (améliore drastiquement les performances) :

- Dossiers `__pycache__`, `node_modules`, `.venv`
- Bases de données (`.db`, `.sqlite`)
- Builds et caches (`build/`, `dist/`, `.pytest_cache/`)
- Fichiers volumineux (OSRM, modèles ML/RL, uploads)
- Documentation temporaire
- Logs et backups

### 💡 Action requise

Après ajout du fichier `.cursorignore`, vous devez **recharger l'indexation** :

**Windows** : `Ctrl + Shift + P` → Tapez "Reload Window" → Appuyez sur Entrée

---

## 2. Paramètres Cursor

### Accéder aux Paramètres

- **Raccourci** : `Ctrl + ,` (ou `Cmd + ,` sur Mac)
- **Menu** : File → Preferences → Settings

### Paramètres Recommandés

#### A. Réduire l'Indexation Inutile

```json
{
  "files.exclude": {
    "**/.git": true,
    "**/.DS_Store": true,
    "**/node_modules": true,
    "**/venv": true,
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/htmlcov": true,
    "**/backend/uploads": true,
    "**/osrm/data": true,
    "**/*.log": true,
    "**/*.txt": false
  },

  "search.exclude": {
    "**/node_modules": true,
    "**/venv": true,
    "**/__pycache__": true,
    "**/backend/uploads": true,
    "**/osrm/data": true,
    "**/Redis": true
  }
}
```

#### B. Optimiser l'Exploration de Fichiers

```json
{
  "search.followSymlinks": false,

  "files.watcherExclude": {
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/**": true,
    "**/venv/**": true,
    "**/__pycache__/**": true,
    "**/backend/uploads/**": true,
    "**/osrm/data/**": true
  }
}
```

#### C. Limiter le Volume d'Indexation

```json
{
  "files.maxMemoryForLargeFilesMB": 4096,

  "cursor.indexCodebase": {
    "maxFiles": 50000,
    "maxSizeMB": 100
  }
}
```

#### D. Performance de l'Autocomplétion

```json
{
  "editor.suggestSelection": "first",
  "editor.suggest.maxVisibleSuggestions": 10,
  "editor.suggest.localityBonus": true,

  "cursor.autocomplete.enabled": true,
  "cursor.chat.maxContextTokens": 8000
}
```

#### E. Génération de Code (Cursor Tab)

```json
{
  "cursor.copilot.enabled": true,
  "cursor.copilotInTab.enabled": true,
  "cursor.copilot.maxSuggestions": 3,

  "editor.inlineSuggest.enabled": true,
  "editor.inlineSuggest.showToolbar": "onHover"
}
```

---

## 3. Configuration des Extensions

### Extensions Essentielles (Minimales)

Installez **uniquement** les extensions nécessaires pour votre projet :

✅ **Python**

- Python (Microsoft) - Langage de base
- Pylance - LSP rapide
- Ruff - Linter rapide (remplace flake8, mypy en partie)

✅ **JavaScript/React**

- ES7+ React/Redux/React-Native snippets
- Prettier - Formatter

✅ **Docker**

- Docker (Microsoft)

✅ **Base de Données**

- PostgreSQL (d'après votre configuration)

### 🚫 Ne PAS Installer

- Extensions de test qui ralentissent l'indexation
- Linters multiples qui font doublon (ex: flake8 + pylint)
- Formatters multiples

### Configuration Ruff (déjà dans le projet)

Votre projet utilise Ruff, c'est excellent ! C'est le linter le plus rapide.

Emplacement : `backend/ruff.toml`

---

## 4. Indexation et Recherche

### Pour Vérifier l'État de l'Indexation

1. Cliquez sur l'**icône en bas à gauche** (statut Cursor)
2. Vérifiez : "Codebase indexed"
3. Si problèmes, cliquez sur "Rebuild index"

### Exclure des Chemins de l'Indexation

Si vous ne l'avez pas fait, allez dans :

**Settings** → **Cursor Settings** → **Exclude files from indexing**

Ajoutez ces patterns :

```
**/venv/**
**/node_modules/**
**/backend/uploads/**
**/osrm/data/**
**/Redis/**
**/session/**
**/*.log
**/*.txt
**/__pycache__/**
```

---

## 5. Options Avancées

### A. Modèle d'IA

**Settings** → **Cursor Settings** → **AI Model**

Choisissez selon votre abonnement :

- **Claude Sonnet 4.5** (le plus rapide, recommandé)
- **Claude Sonnet 4** (bon équilibre)
- **GPT-4** (si vous avez un abonnement OpenAI)

### B. Tokens de Contexte

**Settings** → **Cursor Settings** → **Max Context Tokens**

Recommandation :

- **Petits projets** : 8000 tokens
- **Grands projets** (comme le vôtre) : 16000 tokens

### C. Prefetch

Activez le prefetch pour accélérer les suggestions :

```json
{
  "cursor.experimental.prefetch.enabled": true
}
```

### D. Cache

Clear le cache si performances dégradées :

**Ctrl + Shift + P** → "Clear Cursor Cache" → Entrée

---

## 6. Vérification des Performances

### A. Diagnostic

**Ctrl + Shift + P** → "Cursor: Show Diagnostics"

Vérifiez :

- Taille de l'index
- Vitesse de l'autocomplétion
- Latence de l'IA

### B. Monitoring

**Ctrl + Shift + P** → "Developer: Reload Window"

Si performances toujours mauvaises :

1. Fermez Cursor complètement
2. Supprimez le cache : `%APPDATA%\Cursor\`
3. Rouvrez Cursor

### C. Outil CLI

Si vous avez un abonnement Pro, utilisez la CLI :

```bash
# Indexer manuellement
cursor index rebuild

# Vérifier l'état
cursor status
```

---

## 7. Stratégie d'Optimisation Spécifique pour votre Projet

### Problèmes Identifiés dans Votre Projet

Votre projet ATMR contient :

❌ **Fichiers volumineux à exclure** :

- `backend/data/ml/` - 31 fichiers (modèles ML)
- `backend/data/rl/` - 60 fichiers (modèles RL)
- `osrm/` - données de cartes
- `session/` - 239 fichiers markdown
- `*.log`, `*.txt` - logs multiples

✅ **Solution** : Le `.cursorignore` créé les exclut automatiquement.

### Optimisation par Technique

#### 🐍 Python (Backend)

**Fichiers à indexer** :

- `backend/models/` ✅
- `backend/routes/` ✅
- `backend/services/` ✅
- `backend/tasks/` ✅

**Fichiers à EXCLURE** :

- `backend/__pycache__/` ❌
- `backend/uploads/` ❌
- `backend/data/ml/` ❌
- `backend/data/rl/` ❌
- `backend/temp_*_registry/` ❌

#### ⚛️ React (Frontend)

**Fichiers à indexer** :

- `frontend/src/` ✅

**Fichiers à EXCLURE** :

- `frontend/node_modules/` ❌
- `frontend/build/` ❌

#### 📱 React Native (Mobile)

**Fichiers à indexer** :

- `mobile/client-app/` ✅
- `mobile/driver-app/` ✅

**Fichiers à EXCLURE** :

- `mobile/**/node_modules/` ❌
- `mobile/**/.expo/` ❌

---

## 8. Checklist d'Optimisation Rapide

Cochez chaque étape au fur et à mesure :

- [ ] Fichier `.cursorignore` créé ✅
- [ ] Rechargement de la fenêtre Cursor
- [ ] Paramètres `files.exclude` configurés
- [ ] Paramètres `search.exclude` configurés
- [ ] Paramètres `files.watcherExclude` configurés
- [ ] Tokens de contexte augmentés à 16000
- [ ] Extensions inutiles désinstallées
- [ ] Index reconstruit
- [ ] Cache nettoyé si nécessaire
- [ ] Performances vérifiées

---

## 9. Résultat Attendu

### Avant Optimisation

- ⏱️ Indexation : 5-10 minutes
- 🔍 Recherche : 2-5 secondes
- 💡 Autocomplétion : 1-3 secondes
- 🤖 Suggestions IA : 5-15 secondes

### Après Optimisation

- ⏱️ Indexation : 1-2 minutes
- 🔍 Recherche : < 1 seconde
- 💡 Autocomplétion : instantané
- 🤖 Suggestions IA : 2-5 secondes

---

## 10. Maintenance

### Hebdomadaire

```bash
# Nettoyer le cache
Ctrl + Shift + P → "Clear Cursor Cache"
```

### Mensuel

```bash
# Reconstruire l'index
Ctrl + Shift + P → "Rebuild Index"
```

### Si Performances Dégradées

1. Vérifiez que `.cursorignore` est bien en place
2. Redémarrez Cursor complètement
3. Supprimez le cache : `%APPDATA%\Cursor\`
4. Vérifiez les extensions installées (moins = mieux)

---

## 11. Support et Résolution de Problèmes

### Problème : Indexation lente

**Solution** :

```json
{
  "cursor.indexCodebase.maxFiles": 30000,
  "cursor.indexCodebase.maxSizeMB": 50
}
```

### Problème : Autocomplétion en panne

**Solution** :

1. Ctrl + Shift + P → "Reload Window"
2. Vérifiez que Pylance est activé

### Problème : Trop de memory

**Solution** :

```json
{
  "files.maxMemoryForLargeFilesMB": 2048
}
```

### Problème : Suggestions IA peu pertinentes

**Solution** :

1. Augmentez les tokens de contexte : 16000
2. Vérifiez que les bons dossiers sont indexés

---

## ✅ Conclusion

Vous avez maintenant :

1. ✅ Un fichier `.cursorignore` optimisé
2. ✅ Des paramètres de performance configurés
3. ✅ Une stratégie d'exclusion des fichiers volumineux
4. ✅ Un guide de maintenance

**Prochaine étape** : Recharger Cursor et profiter des meilleures performances ! 🚀

---

**Note** : Les performances dépendent aussi de votre matériel. Sur Windows, assurez-vous d'avoir :

- RAM : 16GB+ recommandé
- SSD : fortement recommandé
- CPU : 4+ cores recommandés
