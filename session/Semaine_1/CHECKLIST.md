# ✅ CHECKLIST SEMAINE 1

**Cochez les cases au fur et à mesure de votre progression.**

---

## 📅 JOUR 1 : Lundi - Fichiers Excel Inutiles

### Matin (1h)

- [ ] Lire le guide Jour 1 complet
- [ ] Vérifier que les fichiers ne sont pas référencés (`grep`)
- [ ] Créer dossier backup `session/backup_semaine1`

### Après-midi (1h)

- [ ] Copier fichiers dans backup
- [ ] Supprimer Classeur1.xlsx
- [ ] Supprimer transport.xlsx
- [ ] Vérifier suppression (`ls -la *.xlsx`)

### Fin de journée (30min)

- [ ] Lancer application (vérifier pas d'erreur)
- [ ] Commit Git avec message approprié
- [ ] Push vers origin/main
- [ ] Remplir rapport quotidien Jour 1

**✅ Validation Jour 1** : Fichiers supprimés, backup créé, commit fait

---

## 📅 JOUR 2 : Mardi - check_bookings.py

### Matin (1h30)

- [ ] Lire le guide Jour 2 complet
- [ ] Lire le contenu de check_bookings.py
- [ ] Rechercher toutes les références (`grep -r "check_bookings"`)
- [ ] Créer backup + README explicatif

### Après-midi (1h)

- [ ] Supprimer check_bookings.py
- [ ] Lancer application (vérifier fonctionnement)
- [ ] Lancer tests (`pytest tests/ -v`)
- [ ] Vérifier logs (aucune erreur)

### Fin de journée (30min)

- [ ] Commit Git
- [ ] Push vers origin/main
- [ ] Remplir rapport quotidien Jour 2

**✅ Validation Jour 2** : Script supprimé, tests OK, application fonctionne

---

## 📅 JOUR 3 : Mercredi - Refactoriser Haversine

### Matin (3h)

- [ ] Lire le guide Jour 3 complet
- [ ] Trouver les 3 implémentations Haversine (`grep`)
- [ ] Créer `backend/shared/geo_utils.py`
- [ ] Créer `backend/tests/test_geo_utils.py`
- [ ] Lancer tests `pytest tests/test_geo_utils.py -v`
- [ ] Tous les 12 tests passent

### Après-midi (3h)

- [ ] Remplacer dans `heuristics.py`
- [ ] Remplacer dans `data.py`
- [ ] Remplacer dans `route_analysis.py`
- [ ] Tests de non-régression (`pytest tests/ -v`)
- [ ] Lancer application complète

### Fin de journée (30min)

- [ ] Commit Git avec tous les fichiers modifiés
- [ ] Push vers origin/main
- [ ] Remplir rapport quotidien Jour 3

**✅ Validation Jour 3** : geo_utils créé, 12 tests passent, 3 fichiers refactorisés

---

## 📅 JOUR 4 : Jeudi - Sérialisation Marshmallow

### Matin (3h)

- [ ] Lire le guide Jour 4 complet
- [ ] Analyser sérialisations existantes (`grep "serialize"`)
- [ ] Installer Marshmallow (`pip install marshmallow`)
- [ ] Ajouter à requirements.txt
- [ ] Créer `backend/schemas/dispatch_schemas.py`
- [ ] Créer `backend/tests/test_dispatch_schemas.py`
- [ ] Lancer tests `pytest tests/test_dispatch_schemas.py -v`

### Après-midi (3h)

- [ ] Remplacer dans `apply.py`
- [ ] Remplacer dans `dispatch_routes.py`
- [ ] Tests de non-régression (`pytest tests/ -v`)
- [ ] Tester API (`curl http://localhost:5000/api/assignments`)

### Fin de journée (30min)

- [ ] Commit Git
- [ ] Push vers origin/main
- [ ] Remplir rapport quotidien Jour 4

**✅ Validation Jour 4** : Schémas créés, 15 tests passent, API fonctionne

---

## 📅 JOUR 5 : Vendredi - Revue et Validation

### Matin (2h)

- [ ] Lire le guide Jour 5 complet
- [ ] Revue code complet (`git diff HEAD~4 HEAD`)
- [ ] Relire tous les fichiers modifiés
- [ ] Vérifier qualité du code
- [ ] Tous les tests unitaires (`pytest tests/ -v --cov`)

### Après-midi (2h)

- [ ] Tests manuels application complète
- [ ] Test dispatch end-to-end
- [ ] Vérifier logs (aucune erreur)
- [ ] Mesurer l'impact (lignes, fichiers, tests)
- [ ] Créer `session/SEMAINE_1_IMPACT.md`
- [ ] Créer `session/SEMAINE_1_RAPPORT.md`

### Fin de journée (1h)

- [ ] Mettre à jour README.md (si nécessaire)
- [ ] Commit final
- [ ] Push vers origin/main
- [ ] Remplir rapport final semaine 1
- [ ] **Célébrer** ! 🎉

**✅ Validation Jour 5** : Revue complète, rapports créés, documentation à jour

---

## 📊 RÉSUMÉ SEMAINE

### Métriques Finales

- [ ] Code supprimé : ~400 lignes ✅
- [ ] Tests ajoutés : 27 tests ✅
- [ ] Fichiers supprimés : 3 ✅
- [ ] Fichiers créés : 4 (+ 2 tests) ✅
- [ ] Coverage : +12% ✅
- [ ] Application fonctionne : ✅

### Livrables

- [ ] `shared/geo_utils.py` créé
- [ ] `schemas/dispatch_schemas.py` créé
- [ ] `tests/test_geo_utils.py` créé (12 tests)
- [ ] `tests/test_dispatch_schemas.py` créé (15 tests)
- [ ] 3 fichiers refactorisés
- [ ] 5 commits Git propres
- [ ] Rapport final complet

### Validation Finale

- [ ] Tous les tests passent (27/27)
- [ ] Application fonctionne normalement
- [ ] Aucune régression détectée
- [ ] Documentation à jour
- [ ] Code propre et lisible
- [ ] Backup créé (rollback possible)

---

## 🎉 SEMAINE 1 COMPLÉTÉE !

**Prochaine étape** : Semaine 2 - Optimisations Base de Données

**Date de début Semaine 2** : \***\*\_\_\_\*\***

**Repos bien mérité ce weekend ! 💪**

---

## 📝 Notes Personnelles

_Espace pour vos notes pendant la semaine :_

**Lundi :**

**Mardi :**

**Mercredi :**

**Jeudi :**

**Vendredi :**

**Difficultés rencontrées :**

**Apprentissages :**

**Idées d'amélioration :**
