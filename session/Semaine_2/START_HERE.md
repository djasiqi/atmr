# 🚀 DÉMARRER ICI - SEMAINE 2

**Bienvenue dans votre dossier Semaine 2 - Optimisations Base de Données !**

Tout est prêt pour vous. Suivez simplement les étapes ci-dessous.

---

## ✅ ÉTAPE 1 : Vérifier que Semaine 1 est Terminée

Avant de commencer la Semaine 2, assurez-vous que :

```bash
cd session/Semaine_1

# Vérifier rapport final existe
cat RAPPORT_FINAL.md

# Vérifier tous les tests passent
cd ../../backend
.\venv\Scripts\python.exe -m pytest tests/test_geo_utils.py tests/test_dispatch_schemas.py -v
```

**Résultat attendu** :

- ✅ 38/38 tests passent
- ✅ Rapport Semaine 1 complété
- ✅ 0 erreur de linter

---

## ✅ ÉTAPE 2 : Ce que Vous Allez Faire Cette Semaine

### Vue d'Ensemble

**Objectif** : Rendre votre base de données **ultra-performante** 🚀

**Problème actuel** :

- Dispatch prend 45 secondes
- 150+ requêtes SQL par dispatch
- Queries lentes (>100ms)
- apply_assignments prend 2.5s

**Après cette semaine** :

- Dispatch en 20 secondes (-56%)
- 50 requêtes SQL (-67%)
- Queries rapides (<50ms)
- apply_assignments en 0.25s (-90%)

---

## ✅ ÉTAPE 3 : Structure du Dossier

Votre dossier contient :

```
session/Semaine_2/
├── START_HERE.md                 ← 🎯 VOUS ÊTES ICI
├── README.md                      ← Point d'entrée principal
├── GUIDE_DETAILLE.md              ← Guide complet jour par jour
├── CHECKLIST.md                   ← Votre to-do list
├── COMMANDES.md                   ← Commandes à copier-coller
├── STRUCTURE.md                   ← Organisation du dossier
│
├── RAPPORT_QUOTIDIEN_TEMPLATE.md  ← Template rapport (x5)
├── RAPPORT_FINAL_TEMPLATE.md      ← Template rapport final
│
└── rapports/                      ← Vos rapports (à créer)
```

---

## ✅ ÉTAPE 4 : Ordre de Lecture Recommandé

### 🥇 Maintenant (30 minutes)

1. **Ce fichier** `START_HERE.md` (5 min) ✅ EN COURS
2. **README.md** (10 min) - Vue d'ensemble
3. **CHECKLIST.md** (5 min) - Voir toutes les tâches
4. **GUIDE Jour 1** (10 min) - Profiling DB

### 🥈 Lundi Matin (1ère heure)

1. **GUIDE_DETAILLE.md** - Section Jour 1 complète
2. **COMMANDES.md** - Commandes du jour
3. **Commencer le profiling !**

---

## ✅ ÉTAPE 5 : Prérequis Techniques

### Outils à Installer

```bash
# 1. pgAdmin (Windows)
# Télécharger : https://www.pgadmin.org/download/
# OU DBeaver : https://dbeaver.io/download/

# 2. SQLite Browser (si SQLite)
# Télécharger : https://sqlitebrowser.org/

# 3. Packages Python nécessaires
cd backend
.\venv\Scripts\python.exe -m pip install sqlalchemy-utils flask-migrate
```

### Backup Base de Données

```bash
# IMPORTANT : Backup AVANT toute modification !
cd backend

# Si PostgreSQL
pg_dump -U postgres -d atmr_db > ../session/backup_semaine2/db_backup.sql

# Si SQLite
cp instance/development.db ../session/backup_semaine2/development.db.backup
```

---

## 🎯 VOUS ÊTES PRÊT !

### Prochaine Action

**MAINTENANT : Lire README.md**

```bash
cd session/Semaine_2
code README.md
```

**Puis suivre les instructions !**

---

## 📊 Ce que Vous Allez Accomplir

### Jour 1 (Lundi) - Profiling

- Identifier les 10 queries les plus lentes
- Analyser les goulots d'étranglement
- Créer rapport de profiling

### Jour 2 (Mardi) - Index DB

- Créer migration Alembic
- Ajouter 5-10 index manquants
- Tester performance avant/après

### Jour 3 (Mercredi) - Bulk Inserts

- Refactoriser apply.py
- Remplacer boucles par bulk_insert_mappings
- -90% temps d'écriture DB

### Jour 4 (Jeudi) - Queries N+1

- Identifier queries N+1
- Ajouter joinedload/selectinload
- Éliminer requêtes inutiles

### Jour 5 (Vendredi) - Validation

- Benchmarks avant/après
- Tests performance
- Rapport final

---

## ⚡ DÉMARRAGE ULTRA-RAPIDE

**Vous avez 5 minutes ?**

1. **Lire README** et voir planning
2. **Ouvrir CHECKLIST** et voir Jour 1
3. **Faire backup DB** (CRITIQUE !)
4. **Commencer !**

---

## 📝 Checklist Avant de Commencer

- [ ] Semaine 1 terminée et validée
- [ ] Tous les fichiers Semaine 2 créés
- [ ] Backup DB effectué
- [ ] Outils installés (pgAdmin/DBeaver)
- [ ] Git status clean (ou commit Semaine 1)
- [ ] Prêt à travailler ! 💪

---

**Bonne semaine 2 ! Vous allez rendre votre DB ultra-rapide ! ⚡🚀**

---

**Créé le** : 20 octobre 2025  
**Prêt pour** : Semaine 2 - Optimisations DB
