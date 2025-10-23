# 📁 SEMAINE 2 - Optimisations Base de Données

**Période** : Semaine 2 (5 jours)  
**Objectif** : Optimiser les performances de la base de données  
**Livrable** : -50% temps queries, +Performance SQL massive

---

## 📚 Documents de la Semaine

### 📖 Documentation Principale

- **[GUIDE_DETAILLE.md](./GUIDE_DETAILLE.md)** - Guide complet jour par jour (très détaillé)
- **[CHECKLIST.md](./CHECKLIST.md)** - Checklist simple pour suivre votre progression
- **[COMMANDES.md](./COMMANDES.md)** - Toutes les commandes à copier-coller

### 📝 Templates à Remplir

- **[RAPPORT_QUOTIDIEN_TEMPLATE.md](./RAPPORT_QUOTIDIEN_TEMPLATE.md)** - À remplir chaque soir
- **[RAPPORT_FINAL_TEMPLATE.md](./RAPPORT_FINAL_TEMPLATE.md)** - À remplir vendredi soir

### 📊 Résultats (à créer)

- `rapports/jour_1.md` - Votre rapport Jour 1
- `rapports/jour_2.md` - Votre rapport Jour 2
- `rapports/jour_3.md` - Votre rapport Jour 3
- `rapports/jour_4.md` - Votre rapport Jour 4
- `rapports/jour_5.md` - Votre rapport Jour 5
- `RAPPORT_FINAL.md` - Résumé complet de la semaine

---

## 🚀 Démarrage Rapide

### 1️⃣ Avant de Commencer

```bash
# Créer le dossier backup base de données
mkdir -p session/backup_semaine2

# Vérifier que vous êtes sur la bonne branche
git branch
git status

# Backup de la base de données
cd backend
.\venv\Scripts\python.exe manage.py db backup > ../session/backup_semaine2/db_backup.sql
```

### 2️⃣ Chaque Matin

1. Ouvrir **CHECKLIST.md** et voir les tâches du jour
2. Ouvrir **GUIDE_DETAILLE.md** section du jour
3. Ouvrir **COMMANDES.md** pour avoir les commandes à portée

### 3️⃣ Chaque Soir

1. Cocher les tâches terminées dans **CHECKLIST.md**
2. Remplir le rapport quotidien (copier template)
3. Commit Git de vos changements

### 4️⃣ Vendredi Soir

1. Remplir **RAPPORT_FINAL.md**
2. Faire un dernier commit
3. Célébrer ! 🎉

---

## 📅 Planning Semaine

| Jour         | Tâche Principale                | Durée | Fichiers          |
| ------------ | ------------------------------- | ----- | ----------------- |
| **Lundi**    | Analyser queries lentes         | 6h    | Profiling DB      |
| **Mardi**    | Créer index DB manquants        | 6h    | Migration Alembic |
| **Mercredi** | Bulk inserts dans apply.py      | 6h    | apply.py          |
| **Jeudi**    | Optimiser queries N+1           | 6h    | Routes + Services |
| **Vendredi** | Tests performance et validation | 6h    | Benchmarks        |

---

## ✅ Critères de Succès

À la fin de la semaine, vous devez avoir :

- [ ] Profiling DB effectué (requêtes lentes identifiées)
- [ ] 5-10 index DB créés (migration Alembic)
- [ ] Bulk inserts implémentés dans apply.py
- [ ] Queries N+1 éliminées
- [ ] -50% temps queries critiques
- [ ] Benchmarks avant/après documentés
- [ ] Migration DB testée
- [ ] Rapport final complété

---

## 🎯 Objectifs de Performance

### Métriques Cibles

| Métrique                    | Avant      | Cible     | Gain |
| --------------------------- | ---------- | --------- | ---- |
| **Temps dispatch complet**  | 45s        | 20s       | -56% |
| **Queries lentes (>100ms)** | 15 queries | 3 queries | -80% |
| **Temps apply_assignments** | 2.5s       | 0.25s     | -90% |
| **Requêtes par dispatch**   | 150+       | 50        | -67% |

---

## 🆘 Besoin d'Aide ?

### Problème : "Migration échoue"

**Solution** : Revenir en arrière

```bash
cd backend
.\venv\Scripts\python.exe -m flask db downgrade
```

### Problème : "Tests performance échouent"

**Solution** : Vérifier que la DB est bien indexée

```bash
.\venv\Scripts\python.exe -c "from ext import db; print(db.engine.execute('PRAGMA index_list(assignment)').fetchall())"
```

### Problème : "Bulk insert ne fonctionne pas"

**Solution** : Vérifier la syntaxe SQLAlchemy

```python
# Bon usage
db.session.bulk_insert_mappings(Assignment, assignment_dicts)
db.session.commit()
```

---

## 📞 Contact

- **Tech Lead** : [VOTRE NOM]
- **Question urgente** : [EMAIL/SLACK]
- **Documentation** : Ce dossier `session/Semaine_2/`

---

## 🎯 Prérequis

### Semaine 1 Terminée

Avant de commencer la Semaine 2, assurez-vous que :

- [x] Semaine 1 complétée (geo_utils + schemas)
- [x] Tous les tests Semaine 1 passent
- [x] 0 erreur de linter
- [x] Code commité

### Outils Nécessaires

- [ ] pgAdmin ou DBeaver installé (visualisation DB)
- [ ] Connaissance basique SQL
- [ ] Connaissance Alembic (migrations)

---

## 🎯 Prochaine Étape

**MAINTENANT : Ouvrir START_HERE.md**

```bash
cd session/Semaine_2
code START_HERE.md
```

**Bonne semaine 2 ! 💪**
