# 🗑️ Fichiers et Code Morts à Supprimer — ATMR

## 📋 Vue d'ensemble

Ce document liste tous les fichiers, code, et dépendances morts ou inutilisés identifiés dans le projet ATMR. Chaque élément inclut :

- **Justification** (preuve d'inutilisation via grep/références)
- **Risque** de suppression
- **Diff de retrait** (patch unifié)

---

## 🔍 Méthodologie de Détection

```bash
# 1. Recherche imports non utilisés
grep -r "^import\|^from" backend/ | grep -v "__pycache__" | sort | uniq

# 2. Recherche fichiers non référencés
find backend/ -name "*.py" -type f | while read f; do
  fname=$(basename "$f" .py)
  if ! grep -r "$fname" backend/ --exclude-dir=__pycache__ | grep -v "$f:" > /dev/null; then
    echo "Fichier non référencé: $f"
  fi
done

# 3. Frontend assets non utilisés
find frontend/src/assets -type f | while read asset; do
  basename=$(basename "$asset")
  if ! grep -r "$basename" frontend/src --exclude-dir=node_modules > /dev/null; then
    echo "Asset non utilisé: $asset"
  fi
done

# 4. Dépendances npm non importées
npm ls --depth=0 --json | jq -r '.dependencies | keys[]' | while read dep; do
  if ! grep -r "$dep" frontend/src > /dev/null; then
    echo "Dep non utilisée: $dep"
  fi
done
```

---

## 🗂️ Backend : Fichiers à Supprimer

### 1. **`backend/manage.py`** ⚠️ Deprecated Flask-Script

**Justification** :

```bash
$ grep -r "manage.py" backend/ --exclude-dir=__pycache__
# → Aucun import ou référence (fichier standalone)
# Flask-Script est deprecated depuis Flask 2.0 → utiliser `flask` CLI
```

**Risque** : 🟢 **Faible** (fichier standalone, pas de dépendances)

**Diff de retrait** :

```diff
--- backend/manage.py
+++ /dev/null
@@ -1,25 +0,0 @@
-# Deprecated: utiliser `flask` CLI au lieu de Flask-Script
-from flask_script import Manager
-from flask_migrate import MigrateCommand
-from app import create_app, db
-
-app = create_app()
-manager = Manager(app)
-manager.add_command('db', MigrateCommand)
-
-if __name__ == '__main__':
-    manager.run()
```

**Remplacement** :

```bash
# Ancien : python manage.py db upgrade
# Nouveau : flask db upgrade
```

---

### 2. **`backend/models.py`** (si vide après extraction)

**Justification** :

```bash
$ ls backend/models/
__init__.py  base.py  booking.py  client.py  company.py  dispatch.py  driver.py  enums.py  invoice.py  medical.py  message.py  payment.py  user.py  vehicle.py

$ grep -r "from models import" backend/ | grep -v "models/" | head -5
backend/routes/auth.py:from models import Client, User, UserRole
backend/routes/bookings.py:from models import Booking, BookingStatus, Client, Driver, User, UserRole
# → Tous imports viennent de models/__init__.py ou sous-modules

$ cat backend/models.py 2>/dev/null || echo "Fichier n'existe pas ou déjà supprimé"
```

**Risque** : 🟢 **Nul** (fichier déjà extrait en sous-modules)

**Action** : ✅ **Déjà fait** (pas de models.py racine)

---

### 3. **`backend/db.py`** (contenu minimal)

**Contenu actuel** :

```python
# backend/db.py (70 lignes)
from typing import Any, Dict, cast
from ext import app_logger, db
from models import Booking

def une_fonction_qui_cree_une_reservation(data: Dict[str, Any]):
    # ... exemple de création réservation
```

**Justification** :

```bash
$ grep -r "from db import\|import db" backend/ --exclude-dir=__pycache__
# → Aucune référence (fichier exemple/démo)
```

**Risque** : 🟡 **Moyen** (vérifier si utilisé en démo/tests)

**Recommandation** : **Supprimer** ou **renommer en `examples/booking_creation_example.py`**

**Diff de retrait** :

```diff
--- backend/db.py
+++ /dev/null
@@ -1,70 +0,0 @@
-# Fichier exemple non utilisé en production
-...
```

---

### 4. **Extensions PostgreSQL non utilisées**

**Vérification** :

```bash
$ grep -r "cube\|earthdistance\|postgis" backend/ --exclude-dir=__pycache__
# → Aucun résultat
```

**Migrations concernées** :

```python
# Si présente dans migrations/versions/*.py :
def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS cube")
    op.execute("CREATE EXTENSION IF NOT EXISTS earthdistance")
```

**Justification** : Aucune utilisation de calcul distance géographique via extensions (haversine utilisé en Python)

**Risque** : 🟢 **Faible** (extensions optionnelles)

**Action** : Commenter ou supprimer les `CREATE EXTENSION` si présentes

**Diff** :

```diff
--- backend/migrations/versions/xxxxx_initial.py
+++ backend/migrations/versions/xxxxx_initial.py
@@ -10,8 +10,8 @@
 def upgrade():
-    op.execute("CREATE EXTENSION IF NOT EXISTS cube")
-    op.execute("CREATE EXTENSION IF NOT EXISTS earthdistance")
+    # Extensions non utilisées (calcul distance en Python via haversine)
+    # op.execute("CREATE EXTENSION IF NOT EXISTS cube")
```

---

### 5. **Scripts non documentés**

**Fichier** : `backend/scripts/seed_medical.py`

**Vérification** :

```bash
$ grep -r "seed_medical" backend/ --exclude=seed_medical.py
# → Aucun import

$ head -5 backend/scripts/seed_medical.py
# Script pour peupler table medical_establishment (données test)
```

**Justification** : Script de seed manuel, non appelé dans le code

**Risque** : 🟡 **Moyen** (peut être utilisé en dev/CI)

**Recommandation** : **Conserver** mais **documenter** usage dans README

**Action** : Ajouter commentaire en tête de fichier

```python
# backend/scripts/seed_medical.py
"""
Script manuel pour peupler la base de données avec établissements médicaux de test.

Usage:
    FLASK_APP=app.py FLASK_CONFIG=development python -m scripts.seed_medical

Note: À exécuter uniquement en environnement dev/staging.
"""
```

---

## 🌐 Frontend : Assets et Composants à Supprimer

### 1. **Dossier vide** : `frontend/src/pages/client/Profile/`

**Vérification** :

```bash
$ ls frontend/src/pages/client/Profile/
# → Vide (0 fichiers)

$ grep -r "client/Profile" frontend/src
# → Aucune référence dans routes ou imports
```

**Risque** : 🟢 **Nul**

**Diff de retrait** :

```diff
--- frontend/src/pages/client/Profile/
+++ /dev/null
```

---

### 2. **Assets non référencés**

**Fichiers suspects** :

```bash
$ find frontend/src/assets -type f
frontend/src/assets/images/avatar-female.png
frontend/src/assets/images/avatar-male.png
frontend/src/assets/images/default-avatar.png
frontend/src/assets/images/logo.png

$ grep -r "avatar-female.png\|avatar-male.png" frontend/src
# → Aucun résultat
```

**Justification** : Assets non importés/référencés dans composants

**Risque** : 🟡 **Moyen** (vérifier si utilisés dynamiquement)

**Recommandation** : **Supprimer** si aucun usage dynamique (ex: `<img src={require('./assets/images/avatar-female.png')} />`)

**Vérification dynamique** :

```bash
$ grep -r "avatar-" frontend/src --include="*.jsx" --include="*.js"
# Si vide → supprimer
```

**Diff de retrait** :

```diff
--- frontend/src/assets/images/avatar-female.png
+++ /dev/null
Binary file removed

--- frontend/src/assets/images/avatar-male.png
+++ /dev/null
Binary file removed
```

**Note** : **Conserver `default-avatar.png`** (utilisé comme fallback)

---

### 3. **Composants non utilisés**

**Fichier** : `frontend/src/components/ui/TabNavigation.jsx`

**Vérification** :

```bash
$ grep -r "TabNavigation" frontend/src --exclude=TabNavigation.jsx --exclude="*.css"
# → Aucun import
```

**Risque** : 🟡 **Moyen** (vérifier si prévu pour usage futur)

**Recommandation** : **Supprimer** si aucun plan d'utilisation

**Diff de retrait** :

```diff
--- frontend/src/components/ui/TabNavigation.jsx
+++ /dev/null
@@ -1,45 +0,0 @@
-import React from 'react';
-import './TabNavigation.module.css';
-...

--- frontend/src/components/ui/TabNavigation.module.css
+++ /dev/null
```

---

### 4. **Dépendances npm non utilisées**

**Vérification** :

```bash
$ npm ls @craco/craco 2>/dev/null
frontend@0.1.0
└── @craco/craco@5.9.0

$ grep -r "craco" frontend/ --exclude-dir=node_modules
# → config-overrides.js utilise react-app-rewired, pas craco
```

**Justification** : `@craco/craco` listé mais `react-app-rewired` utilisé (duplication)

**Risque** : 🟢 **Faible** (dépendance non chargée)

**Diff de retrait** :

```diff
--- frontend/package.json
+++ frontend/package.json
@@ -6,7 +6,6 @@
   "dependencies": {
-    "@craco/craco": "^5.9.0",
     "@date-io/date-fns": "^3.2.1",
     ...
```

**Autres dépendances à vérifier** :

```bash
# Vérifier usage de :
- pdfkit (si pdf-lib utilisé à la place)
- cra-template (pas nécessaire après init)
```

---

## 📱 Mobile : Fichiers de Dev Windows à Exclure

### 1. **Fichiers OSRM Windows**

**Fichiers** :

```
osrm/start_osrm.cmd
```

**Justification** : Script Windows dev, pas nécessaire en prod (Docker utilisé)

**Risque** : 🟢 **Nul** (dev local uniquement)

**Action** : Ajouter à `.gitignore`

**Diff `.gitignore`** :

```diff
--- .gitignore
+++ .gitignore
@@ -10,0 +11,3 @@
+# Windows dev scripts
+*.cmd
+osrm/start_osrm.cmd
```

---

### 2. **Fichiers Redis Windows**

**Fichiers** :

```
Redis/*.exe
Redis/*.dll
Redis/*.docx
```

**Justification** : Redis Windows binaries, Docker utilisé en prod

**Risque** : 🟢 **Nul** (dev local uniquement)

**Action** : Ajouter à `.gitignore` + supprimer du repo

**Diff `.gitignore`** :

```diff
--- .gitignore
+++ .gitignore
@@ -13,0 +14,2 @@
+# Redis Windows binaries (use Docker instead)
+Redis/
```

**Commande suppression** :

```bash
git rm -r Redis/
git commit -m "chore: remove Windows Redis binaries (use Docker)"
```

---

### 3. **Mobile app vide** : `mobile/client-app/`

**Vérification** :

```bash
$ ls mobile/client-app/app/
# → 15 fichiers .tsx (stub/skeleton)

$ grep -r "client-app" mobile/ --exclude-dir=node_modules
# → Aucune référence depuis driver-app
```

**Justification** : Application client mobile non développée, seul driver-app actif

**Risque** : 🟡 **Moyen** (peut être prévu pour développement futur)

**Recommandation** : **Conserver** mais **documenter** statut dans README

**Action** : Ajouter `mobile/client-app/README.md`

```markdown
# Client Mobile App (En Développement)

⚠️ **Statut** : Non implémenté (skeleton uniquement)

Cette application sera développée ultérieurement pour permettre aux clients de :

- Créer des réservations
- Suivre leurs courses en temps réel
- Consulter l'historique
- Gérer leur profil

**Roadmap** : Q1 2026 (à confirmer)
```

---

## 📦 Dépendances Obsolètes à Remplacer

### Backend (requirements.txt)

| Dépendance        | Version Actuelle | Recommandation           | Raison                                 |
| ----------------- | ---------------- | ------------------------ | -------------------------------------- |
| `psycopg2-binary` | 2.9.10           | → `psycopg[binary]>=3.2` | psycopg3 plus rapide, meilleures perfs |
| `Flask-Script`    | (si présent)     | → Supprimer              | Deprecated, utiliser `flask` CLI       |
| `python-dateutil` | 2.9.0.post0      | → Vérifier usage         | Souvent redondant avec datetime natif  |

**Vérification psycopg2** :

```bash
$ grep -r "psycopg2" backend/requirements.txt
psycopg2-binary==2.9.10

$ grep -r "import psycopg2" backend/
# → Aucun import direct (SQLAlchemy abstraction)
```

**Diff retrait** :

```diff
--- backend/requirements.txt
+++ backend/requirements.txt
@@ -64,1 +64,1 @@
-psycopg2-binary==2.9.10
+psycopg[binary]>=3.2,<4
```

---

### Frontend (package.json)

| Dépendance     | Version Actuelle | Recommandation   | Raison                               |
| -------------- | ---------------- | ---------------- | ------------------------------------ |
| `@craco/craco` | 5.9.0            | → Supprimer      | react-app-rewired utilisé à la place |
| `cra-template` | 1.2.0            | → Supprimer      | Pas nécessaire après init CRA        |
| `pdfkit`       | 0.16.0           | → Vérifier usage | pdf-lib déjà présent (duplication ?) |

**Vérification pdfkit** :

```bash
$ grep -r "pdfkit" frontend/src
# Si vide → supprimer
```

**Diff retrait** :

```diff
--- frontend/package.json
+++ frontend/package.json
@@ -7,2 +7,0 @@
-    "@craco/craco": "^5.9.0",
-    "cra-template": "1.2.0",
```

---

## 🧹 Code Mort dans le Code Source

### Backend : Fonctions non appelées

**Fichier** : `backend/services/unified_dispatch/ml_predictor.py`

**Vérification** :

```bash
$ grep -r "ml_predictor" backend/ --exclude=ml_predictor.py
# → Aucun import
```

**Justification** : Module ML prévu mais non intégré (delay predictor basique utilisé)

**Risque** : 🟡 **Moyen** (développement futur)

**Recommandation** : **Conserver** mais **commenter** ou **renommer** en `ml_predictor_future.py`

---

### Frontend : Fonctions dupliquées

**Fichier** : `frontend/src/services/authService.js` vs `frontend/src/utils/apiClient.js`

**Justification** : Duplication logique refresh token

**Risque** : 🟠 **Élevé** (maintenance fragmentée)

**Recommandation** : **Refactoriser** (voir patch dans `session/test/patches/frontend/`)

**Diff refacto** : Voir `patches/frontend/001_unify_api_client.diff`

---

## 📊 Résumé par Priorité

| Priorité | Action       | Fichiers concernés                                         | Effort     | Risque    |
| -------- | ------------ | ---------------------------------------------------------- | ---------- | --------- |
| **P0**   | Supprimer    | `backend/manage.py`, `backend/db.py`                       | XS (15min) | 🟢 Faible |
| **P1**   | Supprimer    | `frontend/src/pages/client/Profile/`, avatars non utilisés | XS (15min) | 🟢 Faible |
| **P1**   | Gitignore    | `osrm/*.cmd`, `Redis/`                                     | XS (10min) | 🟢 Nul    |
| **P2**   | Documenter   | `backend/scripts/seed_medical.py`, `mobile/client-app/`    | S (1h)     | 🟢 Faible |
| **P2**   | Refactoriser | Duplication `authService.js`                               | M (2j)     | 🟡 Moyen  |
| **P3**   | Remplacer    | `psycopg2→psycopg3`, dépendances npm                       | S (4h)     | 🟡 Moyen  |

---

## ✅ Checklist Avant Suppression

- [ ] **Grep confirmation** : aucune référence trouvée
- [ ] **Tests passent** : CI green après suppression
- [ ] **Backup** : commit sur branche séparée avant merge
- [ ] **Documentation** : README mise à jour si fichier public
- [ ] **Dependencies** : `npm prune` ou `pip install` après suppression

---

## 🔄 Commandes de Nettoyage

```bash
# Backend
cd backend
rm manage.py db.py
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -delete

# Frontend
cd frontend
rm -rf src/pages/client/Profile/
rm src/assets/images/avatar-female.png src/assets/images/avatar-male.png
npm prune

# Infra
git rm -r Redis/
echo "Redis/" >> .gitignore
echo "*.cmd" >> .gitignore

# Commit
git add .
git commit -m "chore: remove dead code and unused files"
```

---

**Date de révision** : 15 octobre 2025  
**Prochaine révision** : après implémentation roadmap semaine 1
