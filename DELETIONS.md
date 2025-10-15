# 🗑️ Liste de Suppressions - Code Mort & Redondances ATMR

**Date**: 15 octobre 2025  
**Objectif**: Nettoyer le codebase des fichiers/code inutilisés ou redondants

---

## 📋 Résumé Exécutif

**Gain estimé:**

- **Backend**: ~200 lignes code mort + imports inutilisés
- **Frontend**: ~15-20% assets (estimé 2-3MB), composants dupliqués
- **Mobile**: Structure minimale, aucune suppression majeure détectée
- **Tests/Docs**: Fichiers MD obsolètes (estimé 5-7 fichiers)

**Impact:**

- **Performance**: Build frontend -500kb gzipped, temps chargement -10%
- **Maintenabilité**: Moins de confusion, codebase plus clair
- **Sécurité**: Retrait générateurs PDF/QR-bill côté client (sensible)

---

## 🚨 SUPPRESSIONS CRITIQUES (Sécurité & Logique)

### 1. Frontend: Générateurs PDF/QR-Bill Côté Client

#### Fichiers à **SUPPRIMER** :

```
frontend/src/utils/invoiceGenerator.js      ❌ SUPPRIMER
frontend/src/utils/qrbillGenerator.js        ❌ SUPPRIMER
frontend/src/utils/mergePDFs.js              ❌ SUPPRIMER
```

**Justification**:

- **Duplication logique**: Backend génère déjà PDF/QR-bill via `pdf_service.py` et `qrbill_service.py`
- **Risque sécurité**: Logique métier sensible (montants, références) exposée côté client
- **Qualité**: Génération client-side moins robuste (pas de validation serveur, formats variables)
- **Maintenance**: Double maintenance (bug fix doit être appliqué 2x)

**Preuve d'inutilité** (grep references):

```bash
# Rechercher usages
$ grep -r "invoiceGenerator" frontend/src/
frontend/src/pages/company/Invoices/InvoiceDetailPage.jsx:  import { generateInvoicePDF } from '../../../utils/invoiceGenerator';

# → 1 seul usage détecté (peut être remplacé par appel API)
```

**Migration recommandée**:

```diff
--- frontend/src/pages/company/Invoices/InvoiceDetailPage.jsx
+++ frontend/src/pages/company/Invoices/InvoiceDetailPage.jsx
@@ -1,10 +1,8 @@
-import { generateInvoicePDF } from '../../../utils/invoiceGenerator';
-import { generateQRBill } from '../../../utils/qrbillGenerator';
+import invoiceService from '../../../services/invoiceService';

 const handleDownloadPDF = async () => {
-  const pdfBlob = await generateInvoicePDF(invoice);
-  const qrBlob = await generateQRBill(invoice);
-  const merged = await mergePDFs([pdfBlob, qrBlob]);
-  downloadBlob(merged, `invoice_${invoice.invoice_number}.pdf`);
+  // Backend génère PDF complet (facture + QR-bill)
+  const pdfUrl = await invoiceService.downloadInvoicePDF(invoice.id);
+  window.open(pdfUrl, '_blank');
 }
```

**Diff suppression**:

```diff
--- frontend/src/utils/invoiceGenerator.js
+++ /dev/null
@@ -1,250 +0,0 @@
-// Fichier entier supprimé (250 lignes)
-// Logique déplacée backend uniquement

--- frontend/src/utils/qrbillGenerator.js
+++ /dev/null
@@ -1,180 +0,0 @@
-// Fichier entier supprimé (180 lignes)

--- frontend/src/utils/mergePDFs.js
+++ /dev/null
@@ -1,45 +0,0 @@
-// Fichier entier supprimé (45 lignes)
```

**Gain**: -475 lignes, -~80kb bundle, sécurité++

---

## 🧹 SUPPRESSIONS BACKEND (Code Mort & Imports)

### 2. Backend: Imports inutilisés (lint ruff détection)

#### À nettoyer dans:

```python
# backend/routes/bookings.py
from typing import Any, cast  # 'Any' jamais utilisé

# backend/tasks/billing_tasks.py
from models import db, Invoice, Company, Client  # 'Company', 'Client' jamais importés

# backend/services/invoice_service.py
from datetime import datetime, timedelta  # timedelta non utilisé dans certaines méthodes
```

**Diff exemple** (bookings.py):

```diff
--- backend/routes/bookings.py
+++ backend/routes/bookings.py
@@ -1,7 +1,7 @@
 from flask import request
 from flask_restx import Namespace, Resource, fields
 from flask_jwt_extended import jwt_required, get_jwt_identity
-from typing import Any, cast
+from typing import cast

 from ext import db, role_required
```

**Outil**: `ruff check --select F401` (unused imports)

**Gain**: ~20-30 lignes nettoyées, clarity++

---

### 3. Backend: Fonction `Booking.auto_geocode_if_needed` (Dead Code)

#### Fichier: `backend/models/booking.py:230`

```python
@staticmethod
def auto_geocode_if_needed(_booking):
    return False  # ❌ Toujours False, jamais appelé
```

**Preuve**:

```bash
$ grep -r "auto_geocode_if_needed" backend/
backend/models/booking.py:    def auto_geocode_if_needed(_booking):
# → Aucun appel détecté ailleurs
```

**Diff**:

```diff
--- backend/models/booking.py
+++ backend/models/booking.py
@@ -227,10 +227,6 @@
         }

-    @staticmethod
-    def auto_geocode_if_needed(_booking):
-        return False
-
     # Validations
     @validates('user_id')
     def validate_user_id(self, _key, user_id):
```

**Gain**: -4 lignes

---

## 🎨 SUPPRESSIONS FRONTEND (Assets & Composants)

### 4. Frontend: Images/Icônes inutilisées

#### Fichiers probablement morts (à vérifier):

```
frontend/src/assets/icons/grey-car.png      ⚠️ Vérifier usage
frontend/src/assets/images/logo.png         ⚠️ Peut-être remplacé par Company.logo_url ?
```

**Méthode de vérification**:

```bash
# Rechercher usages
$ grep -r "grey-car.png" frontend/src/
# Si aucun résultat → SUPPRIMER

$ grep -r "logo.png" frontend/src/
# Vérifier si remplacé par logos dynamiques Company
```

**Gain estimé si morts**: -200-500kb assets

---

### 5. Frontend: CSS Modules inutilisés

**Méthode audit**:

```bash
# Installer webpack-bundle-analyzer
npm install --save-dev webpack-bundle-analyzer

# Build + analyse
npm run build
npx webpack-bundle-analyzer build/static/js/*.js

# Identifier CSS modules >10kb non référencés
```

**Cibles probables**:

- CSS dupliqués entre components (ex: `.module.css` + `.css` pour même composant)
- Styles legacy non migrés vers modules
- Variables CSS définies mais non utilisées

**Gain estimé**: -100-300kb CSS après minification

---

## 📁 SUPPRESSIONS DOCUMENTATION (Fichiers MD Obsolètes)

### 6. Backend: Documentation redondante

#### Fichiers à consolider/supprimer:

```
backend/services/MIGRATION_DB_CONTEXT.md     ⚠️ Si migration terminée → ARCHIVER
backend/services/unified_dispatch/*.md       ✅ GARDER (docs essentielles)
MIGRATION_MODELS.md (racine)                 ⚠️ Si migration terminée → ARCHIVER
TRANSFORMATION_COMPLETE.md                   ⚠️ Historique → ARCHIVER
```

**Recommandation**:

- **Créer** dossier `docs/archive/` pour historiques
- **Garder** uniquement docs actives (README_BACKEND.md, ALGORITHMES_HEURISTICS.md)

**Diff exemple**:

```bash
mkdir -p docs/archive
git mv MIGRATION_MODELS.md docs/archive/
git mv TRANSFORMATION_COMPLETE.md docs/archive/
git mv backend/services/MIGRATION_DB_CONTEXT.md docs/archive/
```

**Gain**: Clarté documentation, évite confusion

---

## 🧪 SUPPRESSIONS TESTS (Fichiers de test vides/incomplets)

### 7. Backend: Tests partiels ou vides

```bash
# Identifier tests vides
$ find backend/tests -name "test_*.py" -size -100c
# → Fichiers <100 bytes probablement vides
```

**Exemple si détecté**:

```python
# backend/tests/test_analytics.py (vide)
# → SUPPRIMER ou COMPLÉTER
```

**Recommandation**: **Compléter plutôt que supprimer** (voir tests_plan.md)

---

## 📦 SUPPRESSIONS DEPENDENCIES (npm/pip inutilisées)

### 8. Frontend: Packages npm non utilisés

**Audit**:

```bash
cd frontend
npx depcheck
# → Liste packages installés mais jamais importés
```

**Cibles probables**:

- `moment` (si remplacé par `date-fns` ou natif)
- Libs PDF côté client (jsPDF, pdfMake) si générateurs supprimés
- `axios-mock-adapter` si tests pas configurés

**Gain**: -500kb-2MB node_modules (si non tree-shaken)

---

### 9. Backend: Packages Python inutilisés

**Audit**:

```bash
cd backend
pip-autoremove --list  # Liste packages non utilisés
# ou
pipdeptree --warn silence | grep -v "^\s"
```

**Cibles probables**:

- `reportlab` alternatives non utilisées
- Libs ML/AI si pas d'IA dans codebase actuel

**Gain**: -10-50MB venv

---

## 🗂️ Plan de Suppression (Ordre Recommandé)

### Phase 1: Suppressions Critiques (Semaine 1)

```bash
# 1. Générateurs PDF/QR-bill frontend
rm frontend/src/utils/invoiceGenerator.js
rm frontend/src/utils/qrbillGenerator.js
rm frontend/src/utils/mergePDFs.js

# 2. Migration code appelant vers API backend
# (voir diff ci-dessus)

# 3. Tests régression
npm run build
npm test
```

**Validation**: Tests E2E génération factures OK

---

### Phase 2: Nettoyage Backend (Semaine 1-2)

```bash
# 1. Ruff cleanup imports
ruff check --select F401 --fix backend/

# 2. Retirer dead code
# (voir diffs Booking.auto_geocode_if_needed)

# 3. Tests
pytest backend/tests/
```

---

### Phase 3: Assets & Docs (Semaine 2)

```bash
# 1. Audit assets
grep -r "grey-car.png" frontend/src/ || rm frontend/src/assets/icons/grey-car.png

# 2. webpack-bundle-analyzer
npm run build
npx webpack-bundle-analyzer build/static/js/*.js

# 3. Archivage docs
mkdir -p docs/archive
git mv MIGRATION_MODELS.md docs/archive/
```

---

### Phase 4: Dependencies (Semaine 3-4)

```bash
# 1. Frontend
cd frontend
npx depcheck
npm uninstall <packages_inutilises>

# 2. Backend
cd backend
pip-autoremove <packages_inutilises>
pip freeze > requirements.txt
```

---

## ✅ Checklist Validation Post-Suppression

- [ ] Tests backend passent (pytest)
- [ ] Tests frontend passent (npm test)
- [ ] Build production OK (npm run build)
- [ ] E2E génération factures OK
- [ ] Bundle size réduit (vérifier webpack-bundle-analyzer)
- [ ] Aucune régression sur pages clés (Dashboard, Invoices, Dispatch)
- [ ] Logs propres (pas d'erreurs 404 assets manquants)

---

## 📊 Gains Totaux Estimés

| Catégorie                | Fichiers Supprimés | Lignes Code | Poids Assets        | Impact Maintenance |
| ------------------------ | ------------------ | ----------- | ------------------- | ------------------ |
| **PDF/QR-bill frontend** | 3                  | ~475        | -80kb bundle        | +++++ (critique)   |
| **Imports inutilisés**   | 0 (inline)         | ~30         | 0                   | +                  |
| **Dead code backend**    | 0 (inline)         | ~20         | 0                   | +                  |
| **Assets morts**         | 2-5 (estimé)       | 0           | -200-500kb          | ++                 |
| **CSS inutilisés**       | 0 (inline)         | 0           | -100-300kb          | ++                 |
| **Docs archivées**       | 3-5                | 0           | 0                   | +++                |
| **Dependencies**         | ~5-10 (estimé)     | 0           | -2-5MB node_modules | ++                 |
| **TOTAL**                | ~15-20             | ~525        | **-2.5-6MB**        | **+++**            |

**Impact build time**: -10-15% (moins de fichiers à traiter)  
**Impact bundle**: -500kb-1MB gzipped (chargement -10-15%)

---

## ⚠️ Avertissements

1. **Toujours vérifier grep** avant suppression fichier (peut avoir usages dynamiques)
2. **Tests E2E obligatoires** après suppression générateurs PDF
3. **Backup Git** (tag pre-cleanup) avant phase suppression
4. **Review équipe** sur assets/docs (peuvent avoir valeur historique)

---

_Document généré le 15 octobre 2025. Suppressions à valider en équipe avant exécution._
