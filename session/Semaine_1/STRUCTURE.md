# 📁 STRUCTURE DOSSIER SEMAINE 1

Voici l'organisation complète du dossier pour la Semaine 1.

---

## 🗂️ Arborescence

```
session/Semaine_1/
│
├── README.md                           ⭐ COMMENCER ICI
├── GUIDE_DETAILLE.md                   📖 Guide complet jour par jour
├── CHECKLIST.md                        ✅ Checklist de progression
├── COMMANDES.md                        🖥️ Toutes les commandes
├── STRUCTURE.md                        📁 Ce fichier
│
├── RAPPORT_QUOTIDIEN_TEMPLATE.md       📝 Template à copier chaque soir
├── RAPPORT_FINAL_TEMPLATE.md           📊 Template rapport final
│
└── rapports/                           📂 Vos rapports quotidiens
    ├── .gitkeep
    ├── jour_1.md                       (À créer lundi soir)
    ├── jour_2.md                       (À créer mardi soir)
    ├── jour_3.md                       (À créer mercredi soir)
    ├── jour_4.md                       (À créer jeudi soir)
    └── jour_5.md                       (À créer vendredi soir)
```

---

## 📚 Description des Fichiers

### 📖 Documentation Principale

#### `README.md` ⭐

**C'est votre point d'entrée !**

- Vue d'ensemble de la semaine
- Démarrage rapide
- Planning jour par jour
- Critères de succès
- FAQ

**Quand l'utiliser** : Le lundi matin en premier

---

#### `GUIDE_DETAILLE.md` 📖

**Le guide complet et ultra-détaillé.**

- Explication de chaque étape
- Code à écrire
- Commandes à lancer
- Validation à chaque étape
- ~1000 lignes de contenu

**Quand l'utiliser** : Tout au long de la semaine, section par section

---

#### `CHECKLIST.md` ✅

**Votre to-do list de la semaine.**

- Cases à cocher
- Progression visuelle
- Validation par jour
- Notes personnelles

**Quand l'utiliser** :

- Chaque matin : voir tâches du jour
- Tout au long de la journée : cocher au fur et à mesure
- Chaque soir : vérifier validation

---

#### `COMMANDES.md` 🖥️

**Toutes les commandes prêtes à copier-coller.**

- Organisées par jour
- Copy-paste direct
- Commandes d'urgence
- Debugging

**Quand l'utiliser** : Quand vous avez besoin d'une commande spécifique

---

### 📝 Templates

#### `RAPPORT_QUOTIDIEN_TEMPLATE.md`

**Template pour vos rapports de fin de journée.**

Contient :

- Objectif du jour
- Tâches réalisées
- Résultats (métriques)
- Apprentissages
- Problèmes rencontrés
- Auto-évaluation

**Comment utiliser** :

```bash
# Copier le template
cp RAPPORT_QUOTIDIEN_TEMPLATE.md rapports/jour_1.md

# Éditer
code rapports/jour_1.md
# OU
nano rapports/jour_1.md
```

---

#### `RAPPORT_FINAL_TEMPLATE.md`

**Template pour le rapport final du vendredi.**

Contient :

- Résumé exécutif
- Détails par jour
- Métriques finales
- Objectifs vs Résultats
- Livrables
- Apprentissages
- Auto-évaluation
- Validation

**Comment utiliser** :

```bash
# Vendredi soir
cp RAPPORT_FINAL_TEMPLATE.md RAPPORT_FINAL.md
code RAPPORT_FINAL.md
```

---

### 📂 Dossier Rapports

#### `rapports/`

**Contient vos 5 rapports quotidiens.**

**Structure attendue** :

```
rapports/
├── jour_1.md    ← Lundi soir
├── jour_2.md    ← Mardi soir
├── jour_3.md    ← Mercredi soir
├── jour_4.md    ← Jeudi soir
└── jour_5.md    ← Vendredi soir
```

**Ces fichiers seront créés par vous chaque soir.**

---

## 🚀 Workflow Recommandé

### Lundi Matin (Début Semaine)

```bash
cd session/Semaine_1

# 1. Lire README
cat README.md

# 2. Ouvrir CHECKLIST
code CHECKLIST.md

# 3. Ouvrir GUIDE_DETAILLE (section Jour 1)
code GUIDE_DETAILLE.md

# 4. Ouvrir COMMANDES (référence)
code COMMANDES.md
```

### Chaque Jour (Travail)

```bash
# Matin
# 1. Ouvrir CHECKLIST : voir tâches du jour
# 2. Ouvrir GUIDE_DETAILLE : lire section du jour
# 3. Travailler en suivant le guide

# Tout au long de la journée
# 1. Cocher tâches dans CHECKLIST
# 2. Copier-coller commandes depuis COMMANDES.md
# 3. Suivre étapes dans GUIDE_DETAILLE

# Soir
# 1. Copier template rapport quotidien
cp RAPPORT_QUOTIDIEN_TEMPLATE.md rapports/jour_X.md

# 2. Remplir le rapport
code rapports/jour_X.md

# 3. Commit Git
git add rapports/jour_X.md
git commit -m "docs: rapport jour X"
```

### Vendredi Soir (Fin Semaine)

```bash
# 1. Copier template rapport final
cp RAPPORT_FINAL_TEMPLATE.md RAPPORT_FINAL.md

# 2. Remplir rapport final
code RAPPORT_FINAL.md

# 3. Revue complète
# - Vérifier CHECKLIST (tout coché ?)
# - Relire tous rapports quotidiens
# - Vérifier objectifs atteints

# 4. Commit final
git add RAPPORT_FINAL.md
git commit -m "docs: rapport final Semaine 1 - TERMINÉE ✅"
git push origin main

# 5. Célébrer ! 🎉
```

---

## 📱 Raccourcis Utiles

### Ouvrir Tous les Docs (VS Code)

```bash
cd session/Semaine_1
code README.md GUIDE_DETAILLE.md CHECKLIST.md COMMANDES.md
```

### Ouvrir dans Navigateur (Markdown Preview)

Si vous avez une extension Markdown :

- Ouvrir VS Code
- Installer "Markdown Preview Enhanced"
- Ctrl+Shift+V pour preview

### Impression (Optionnel)

Si vous voulez imprimer :

```bash
# Convertir en PDF (si pandoc installé)
pandoc GUIDE_DETAILLE.md -o GUIDE_DETAILLE.pdf
pandoc CHECKLIST.md -o CHECKLIST.pdf
```

---

## 💡 Conseils d'Utilisation

### Pour les Débutants

**Ordre de lecture** :

1. `README.md` (5 min) - Vue d'ensemble
2. `CHECKLIST.md` (3 min) - Voir toutes les tâches
3. `GUIDE_DETAILLE.md` (15 min) - Lire introduction + Jour 1
4. `COMMANDES.md` (2 min) - Parcourir rapidement

**Puis commencer à travailler !**

### Pour les Expérimentés

**Lecture rapide** :

1. `CHECKLIST.md` - Voir tâches
2. `COMMANDES.md` - Copy-paste commandes
3. `GUIDE_DETAILLE.md` - Référence si besoin

**Se référer au guide seulement si bloqué.**

### Pour les Managers

**Suivi de progression** :

1. `CHECKLIST.md` - Voir progression (cases cochées)
2. `rapports/jour_X.md` - Rapports quotidiens
3. `RAPPORT_FINAL.md` - Synthèse complète

---

## 🔧 Maintenance

### Ajouter un Fichier

```bash
cd session/Semaine_1
touch NOUVEAU_FICHIER.md
```

### Modifier un Template

```bash
code RAPPORT_QUOTIDIEN_TEMPLATE.md
# Faire modifications
# Tous les futurs rapports utiliseront la nouvelle version
```

### Backup

```bash
# Backup complet dossier
cp -r session/Semaine_1 session/backup_semaine_1

# OU avec tar
tar -czf semaine_1_backup.tar.gz session/Semaine_1/
```

---

## 📊 Métriques Attendues

À la fin de la semaine, ce dossier doit contenir :

- [x] 7 fichiers documentation (✅ déjà créés)
- [ ] 5 rapports quotidiens (à créer par vous)
- [ ] 1 rapport final (à créer vendredi)
- [ ] CHECKLIST complétée (100%)

**Total** : ~13 fichiers

---

## 🆘 Troubleshooting

### "Je ne vois pas le dossier rapports/"

```bash
# Créer si manquant
mkdir -p session/Semaine_1/rapports
```

### "Erreur quand j'ouvre un .md"

**Solution** : Installer un éditeur Markdown

- VS Code + extension "Markdown All in One"
- Typora (éditeur dédié)
- MarkText (open source)

### "Les commandes ne marchent pas"

**Solution** : Vérifier que vous êtes dans le bon dossier

```bash
pwd  # Afficher dossier actuel
cd C:\Users\jasiq\atmr  # Aller à la racine projet
```

---

## 📞 Support

**Questions ?**

- Relire `README.md` (section FAQ)
- Chercher dans `GUIDE_DETAILLE.md` (Ctrl+F)
- Vérifier `COMMANDES.md` (commandes urgences)

**Tech Lead** : [VOTRE NOM]  
**Contact** : [EMAIL/SLACK]

---

## 🎯 Objectif Final

**À la fin de la semaine, vous aurez :**
✅ Un dossier complet et bien organisé  
✅ 5 rapports quotidiens documentés  
✅ 1 rapport final synthétique  
✅ Une trace complète de votre travail  
✅ Une base pour la Semaine 2

**Bonne semaine ! 🚀**

---

**Dernière mise à jour** : 20 octobre 2025  
**Version** : 1.0
