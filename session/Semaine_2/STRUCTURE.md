# 📁 STRUCTURE DOSSIER SEMAINE 2

Voici l'organisation complète du dossier pour la Semaine 2.

---

## 🗂️ Arborescence

```
session/Semaine_2/
│
├── START_HERE.md                    🎯 POINT D'ENTRÉE
├── README.md                         📖 Vue d'ensemble
├── GUIDE_DETAILLE.md                 📚 Guide complet jour par jour
├── CHECKLIST.md                      ✅ To-do list
├── COMMANDES.md                      🖥️ Toutes les commandes
├── STRUCTURE.md                      📁 Ce fichier
│
├── RAPPORT_QUOTIDIEN_TEMPLATE.md     📝 Template quotidien
├── RAPPORT_FINAL_TEMPLATE.md         📊 Template final
│
└── rapports/                         📂 Vos rapports (à créer)
    ├── jour_1.md
    ├── jour_2.md
    ├── jour_3.md
    ├── jour_4.md
    └── jour_5.md
```

---

## 📚 Description des Fichiers

### 🎯 START_HERE.md

**Votre point d'entrée principal.**

- Vérification Semaine 1 terminée
- Ordre de lecture
- Prérequis techniques
- Backup DB

**Lire en premier** : ⭐⭐⭐⭐⭐

### 📖 README.md

**Vue d'ensemble de la semaine.**

- Planning jour par jour
- Objectifs de performance
- Critères de succès

**Lire après START_HERE** : ⭐⭐⭐⭐⭐

### 📚 GUIDE_DETAILLE.md

**Guide ultra-complet avec code.**

- Jour 1 : Profiling DB
- Jour 2 : Index DB
- Jour 3 : Bulk inserts
- Jour 4 : Queries N+1
- Jour 5 : Validation

**Référence pendant travail** : ⭐⭐⭐⭐⭐

### ✅ CHECKLIST.md

**Votre to-do list de la semaine.**

- Cases à cocher par jour
- Métriques finales
- Notes personnelles

**Utiliser quotidiennement** : ⭐⭐⭐⭐⭐

### 🖥️ COMMANDES.md

**Toutes les commandes SQL, Python, etc.**

- Backup DB
- Migrations
- Profiling
- Benchmarks

**Référence constante** : ⭐⭐⭐⭐⭐

---

## 🚀 Workflow Recommandé

### Lundi Matin

```bash
cd session/Semaine_2

# 1. Lire START_HERE
code START_HERE.md

# 2. Lire README
code README.md

# 3. Ouvrir CHECKLIST
code CHECKLIST.md

# 4. Lire GUIDE Jour 1
code GUIDE_DETAILLE.md
# Chercher "JOUR 1"

# 5. Ouvrir COMMANDES
code COMMANDES.md

# 6. Backup DB !
cd ../../backend
cp instance/development.db ../session/backup_semaine2/db_backup.db

# 7. Commencer Profiling !
```

---

## 🎯 Objectifs Semaine 2

À la fin de la semaine :

- [ ] Temps dispatch : 45s → 20s
- [ ] Queries : 150 → 50
- [ ] Queries lentes : 15 → 3
- [ ] 5-10 index DB créés
- [ ] Bulk inserts implémentés
- [ ] Queries N+1 éliminées
- [ ] Rapport performance complet

---

**Bonne semaine 2 ! ⚡🚀**
