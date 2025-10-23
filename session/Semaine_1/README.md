# 📁 SEMAINE 1 - Nettoyage Code

**Période** : Jour 1 à Jour 5  
**Objectif** : Nettoyer le code mort et améliorer la maintenabilité  
**Livrable** : -10% code inutile, +20% maintenabilité

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
# Créer le dossier rapports
mkdir -p session/Semaine_1/rapports

# Créer le dossier backup
mkdir -p session/backup_semaine1

# Vérifier que vous êtes sur la bonne branche
git branch
git status
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

| Jour         | Tâche Principale            | Durée | Fichiers                 |
| ------------ | --------------------------- | ----- | ------------------------ |
| **Lundi**    | Supprimer fichiers inutiles | 2h    | Excel, check_bookings.py |
| **Mardi**    | Supprimer check_bookings.py | 3h    | check_bookings.py        |
| **Mercredi** | Refactoriser Haversine      | 6h    | geo_utils.py             |
| **Jeudi**    | Centraliser sérialisation   | 6h    | dispatch_schemas.py      |
| **Vendredi** | Revue et validation         | 4h    | Tous                     |

---

## ✅ Critères de Succès

À la fin de la semaine, vous devez avoir :

- [ ] Supprimé 3 fichiers inutiles
- [ ] Créé `shared/geo_utils.py` avec tests
- [ ] Créé `schemas/dispatch_schemas.py` avec tests
- [ ] Ajouté 27 tests unitaires
- [ ] Tous les tests passent
- [ ] Application fonctionne normalement
- [ ] 5 commits Git propres
- [ ] Rapport final complété

---

## 🆘 Besoin d'Aide ?

### Problème : "Les tests ne passent pas"

**Solution** : Vérifier que toutes les dépendances sont installées

```bash
pip install -r requirements.txt
pytest tests/ -v
```

### Problème : "Import error shared.geo_utils"

**Solution** : Créer `__init__.py` si manquant

```bash
touch backend/shared/__init__.py
```

### Problème : "Git conflict"

**Solution** : Stash, pull, pop

```bash
git stash
git pull origin main
git stash pop
```

---

## 📞 Contact

- **Tech Lead** : [VOTRE NOM]
- **Question urgente** : [EMAIL/SLACK]
- **Documentation** : Ce dossier `session/Semaine_1/`

---

## 🎯 Prochaine Étape

**Semaine 2** : Optimisations Base de Données

- Bulk inserts
- Index DB
- Performance queries

**Dossier** : `session/Semaine_2/` (sera créé après)

---

**Bonne semaine ! 💪**
