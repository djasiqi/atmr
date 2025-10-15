# 📦 Archives ATMR

**Date archivage**: 15 Octobre 2025  
**Raison**: Conservation historique, docs actives suffisantes

---

## 📁 Scripts Utilitaires (`scripts/`)

| Fichier | Description | Alternative Actuelle |
|---------|-------------|---------------------|
| `add_admin.py` | Créer administrateur initial (one-off) | `flask cli` ou panneau admin |
| `check_invoices.py` | Diagnostic factures DB | `GET /api/invoices` endpoints |
| `test_monitoring.py` | Tests manuels dispatch | `pytest backend/tests/test_dispatch_*.py` |

**Usage préservé**: Scripts fonctionnels si besoin ponctuel.

### Utilisation des Scripts Archivés

```bash
# Exemple : créer un admin depuis l'archive
cd docs/archive/scripts
python add_admin.py

# Ou depuis la racine
python docs/archive/scripts/add_admin.py
```

---

## 📄 Sessions Historiques (`sessions/`)

| Fichier | Date | Sujet |
|---------|------|-------|
| `JOUR_4_COMPLETE_SUMMARY.md` | 15/10/2025 | Nettoyage + linting |
| `TRANSFORMATION_COMPLETE.md` | 14/10/2025 | Transformation Analytics |
| `REFACTORISATION_BACKEND_COMPLETE.md` | 14-15/10/2025 | Refactoring models 31→14 |
| `PRESENTATION_FINALE.md` | 15/10/2025 | Présentation audit complet |

**Valeur**: Historique détaillé des sessions de travail majeures.

### Contenu des Sessions

- **JOUR_4_COMPLETE_SUMMARY.md**: Rapport journalier incluant suppressions dead code, linting Ruff/ESLint, optimisations
- **TRANSFORMATION_COMPLETE.md**: Transformation du dashboard Analytics (avant/après, graphiques, insights)
- **REFACTORISATION_BACKEND_COMPLETE.md**: Refactorisation complète des models (monolithe 3302 lignes → 14 fichiers modulaires)
- **PRESENTATION_FINALE.md**: Présentation exécutive de l'audit complet ATMR (40 fichiers générés, patches, workflows CI)

---

## 🔍 Références Actives

**Docs principales à consulter** (racine projet):
- `README_AUDIT.md` - Guide démarrage
- `SUMMARY.md` - Résumé exécutif
- `REPORT.md` - Audit technique complet
- `DASHBOARD.md` - Tableau de bord visuel
- `MASTER_INDEX.md` - Navigation complète

**Tests**:
- `tests_plan.md` - Plan tests exhaustif

**Migrations**:
- `MIGRATIONS_NOTES.md` - Migrations DB proposées

---

## 🗂️ Structure Archive

```
docs/archive/
├── README.md (ce fichier)
├── scripts/
│   ├── add_admin.py
│   ├── check_invoices.py
│   └── test_monitoring.py
└── sessions/
    ├── JOUR_4_COMPLETE_SUMMARY.md
    ├── TRANSFORMATION_COMPLETE.md
    ├── REFACTORISATION_BACKEND_COMPLETE.md
    └── PRESENTATION_FINALE.md
```

---

## ⚠️ Notes Importantes

**Scripts archivés** :
- Toujours fonctionnels, mais non recommandés pour usage quotidien
- Alternatives modernes disponibles (voir tableau ci-dessus)
- Conservés pour référence historique et usage exceptionnel

**Docs archivées** :
- Informations intégrées dans docs actives (SUMMARY.md, REPORT.md, etc.)
- Archivées pour traçabilité et historique complet
- Ne pas mettre à jour (snapshot historique figé)

---

**Archivage effectué**: Phase 2 du plan de purge (DELETIONS.md)  
**Réversible via**: `git mv docs/archive/{scripts,sessions}/* .` si besoin

