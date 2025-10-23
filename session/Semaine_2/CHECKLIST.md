# ✅ CHECKLIST SEMAINE 2

**Cochez les cases au fur et à mesure de votre progression.**

---

## 📅 JOUR 1 : Lundi - Profiling Base de Données

### Matin (3h)

- [ ] Installer pgAdmin ou DBeaver
- [ ] Backup complet de la base de données
- [ ] Activer logging SQL dans Flask (echo=True)
- [ ] Installer flask-profiler ou silk

### Après-midi (3h)

- [ ] Lancer dispatch complet avec profiling
- [ ] Identifier les 10 queries les plus lentes
- [ ] Mesurer temps total (baseline)
- [ ] Créer rapport `PROFILING_RESULTS.md`

### Fin de journée (30min)

- [ ] Documenter queries lentes avec EXPLAIN
- [ ] Créer liste priorités (index à ajouter)
- [ ] Remplir rapport quotidien Jour 1

**✅ Validation Jour 1** : Rapport profiling créé, queries lentes identifiées

---

## 📅 JOUR 2 : Mardi - Index Base de Données

### Matin (3h)

- [ ] Créer migration Alembic `add_performance_indexes.py`
- [ ] Ajouter index sur `assignment(booking_id, created_at)`
- [ ] Ajouter index sur `booking(status, scheduled_time, company_id)`
- [ ] Ajouter index sur `driver(company_id, is_available, is_active)`
- [ ] Tester migration (upgrade/downgrade)

### Après-midi (3h)

- [ ] Ajouter index sur `booking(company_id, scheduled_time)`
- [ ] Ajouter index composite `assignment(dispatch_run_id, status)`
- [ ] Appliquer migration en dev
- [ ] Vérifier index créés (PRAGMA index_list / SHOW INDEX)
- [ ] Mesurer performance (benchmark avant/après)

### Fin de journée (30min)

- [ ] Commit migration
- [ ] Documenter gains de performance
- [ ] Remplir rapport quotidien Jour 2

**✅ Validation Jour 2** : 5-10 index créés, migration testée

---

## 📅 JOUR 3 : Mercredi - Bulk Inserts

### Matin (3h)

- [ ] Analyser apply.py (fonction `apply_and_emit`)
- [ ] Identifier boucles avec commits multiples
- [ ] Créer backup de apply.py
- [ ] Refactoriser avec `bulk_insert_mappings()`

### Après-midi (3h)

- [ ] Implémenter bulk insert pour assignments
- [ ] Implémenter bulk update pour bookings (status)
- [ ] Tests unitaires bulk inserts
- [ ] Benchmark avant/après (mesurer gain)

### Fin de journée (30min)

- [ ] Vérifier aucune régression
- [ ] Tests intégration dispatch complet
- [ ] Commit changements
- [ ] Remplir rapport quotidien Jour 3

**✅ Validation Jour 3** : Bulk inserts OK, -90% temps écriture DB

---

## 📅 JOUR 4 : Jeudi - Éliminer Queries N+1

### Matin (3h)

- [ ] Installer flask-sqlalchemy-debug ou nplusone
- [ ] Détecter toutes les queries N+1
- [ ] Lister les endroits problématiques
- [ ] Créer rapport `N_PLUS_ONE_ISSUES.md`

### Après-midi (3h)

- [ ] Ajouter `joinedload()` dans routes/bookings.py
- [ ] Ajouter `selectinload()` dans routes/dispatch_routes.py
- [ ] Refactoriser loops avec queries
- [ ] Tests de non-régression

### Fin de journée (30min)

- [ ] Vérifier nombre de queries réduit
- [ ] Benchmark avant/après
- [ ] Commit changements
- [ ] Remplir rapport quotidien Jour 4

**✅ Validation Jour 4** : Queries N+1 éliminées, -67% requêtes

---

## 📅 JOUR 5 : Vendredi - Tests Performance et Validation

### Matin (3h)

- [ ] Créer script benchmark complet
- [ ] Mesurer performance dispatch (avant/après Semaine 2)
- [ ] Créer graphiques comparatifs
- [ ] Documenter tous les gains

### Après-midi (3h)

- [ ] Tests de charge (100 bookings, 50 drivers)
- [ ] Tests de stress (1000 bookings)
- [ ] Vérifier aucune régression fonctionnelle
- [ ] Tous les tests unitaires

### Fin de journée (1h)

- [ ] Créer `PERFORMANCE_REPORT.md`
- [ ] Mettre à jour README avec résultats
- [ ] Commit final
- [ ] Remplir rapport final semaine 2
- [ ] **Célébrer** ! 🎉

**✅ Validation Jour 5** : Performance validée, rapport complet

---

## 📊 RÉSUMÉ SEMAINE

### Métriques Finales

- [ ] Temps dispatch : 45s → 20s (-56%) ✅
- [ ] Queries par dispatch : 150+ → 50 (-67%) ✅
- [ ] Temps apply : 2.5s → 0.25s (-90%) ✅
- [ ] Index DB : 0 → 10 ✅
- [ ] Queries lentes : 15 → 3 (-80%) ✅

### Livrables

- [ ] Migration Alembic `add_performance_indexes.py`
- [ ] apply.py refactorisé (bulk inserts)
- [ ] Routes optimisées (joinedload)
- [ ] Rapport profiling
- [ ] Rapport performance
- [ ] Benchmarks avant/après

### Validation Finale

- [ ] Migration testée (upgrade + downgrade)
- [ ] Tous les tests passent
- [ ] Application fonctionne normalement
- [ ] Performance gains documentés
- [ ] Backup DB créé (rollback possible)

---

## 🎉 SEMAINE 2 COMPLÉTÉE !

**Prochaine étape** : Semaine 3-4 - ML POC (Proof of Concept)

**Date de début Semaine 3** : \***\*\_\_\_\*\***

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
