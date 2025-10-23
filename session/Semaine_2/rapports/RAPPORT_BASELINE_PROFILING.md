# 📊 RAPPORT BASELINE - PROFILING BASE DE DONNÉES

**Date**: 2025-10-20  
**Semaine**: Semaine 2 - Optimisations Base de Données  
**Tâche**: Lundi - Profiling DB (6h)  
**Responsable**: Investigation initiale DB performance

---

## 🎯 Objectif

Établir une baseline de performance pour le système de dispatch afin de :
1. Identifier les requêtes SQL lentes (>50ms)
2. Mesurer le temps d'exécution total du dispatch
3. Compter le nombre total de requêtes SQL exécutées
4. Créer un point de référence pour les optimisations futures

---

## 🔧 Outils Installés

### 1. **nplusone** (v1.0.0+)
- **Description**: Détecteur de N+1 queries pour SQLAlchemy
- **Installation**: `pip install nplusone`
- **Usage**: Listener SQLAlchemy pour détecter automatiquement les problèmes de N+1

### 2. **Script de Profiling Personnalisé**
- **Fichier**: `backend/scripts/profiling/profile_dispatch.py`
- **Fonctionnalités**:
  - Listeners SQLAlchemy pour mesurer le temps de chaque requête
  - Détection automatique des requêtes >50ms
  - Génération de rapports détaillés (console + fichier)
  - Support SQLite et PostgreSQL (configuration dynamique)

---

## 📈 RÉSULTATS BASELINE

### Exécution du Profiling

**Date d'exécution**: 2025-10-20  
**Company ID**: 1  
**Environment**: Docker (PostgreSQL)

```
======================================================================
PROFILING DISPATCH - DEMARRAGE
======================================================================
Company ID  : 1
Date        : 2025-10-20
Database    : postgresql+psycopg://atmr:atmr@postgres:5432/atmr
======================================================================

======================================================================
RESULTATS PROFILING
======================================================================

Temps total          : 0.10s
Assignments crees    : 0
Total queries SQL    : 15
Queries lentes (>50ms) : 0
```

### Métriques Clés

| Métrique | Valeur | Cible |
|----------|--------|-------|
| **Temps total** | 0.10s | < 1.0s ✅ |
| **Nombre de queries** | 15 | < 50 ✅ |
| **Queries lentes** (>50ms) | 0 | 0 ✅ |
| **Assignments créés** | 0 | N/A |

---

## 🔍 OBSERVATIONS

### ✅ Points Positifs

1. **Performance Excellente**: Temps d'exécution très rapide (100ms)
2. **Aucune Query Lente**: Toutes les requêtes < 50ms
3. **Nombre de Queries Raisonnable**: 15 requêtes pour un cycle de dispatch

### ⚠️ Limitations de la Baseline

1. **Pas de Bookings**: Le test a été effectué sans bookings dans la DB
   - Message système : `[Dispatch] No dispatch possible for company 1: no_bookings`
   - Impact : Les requêtes les plus lourdes (assignments, optimisations) n'ont pas été testées

2. **Données de Test Manquantes**:
   - Pas de drivers actifs
   - Pas de bookings à assigner
   - Pas de calculs OSRM effectués

### 📊 Profil des Requêtes (Estimation)

Les 15 requêtes identifiées sont probablement :
1. Chargement de la configuration Company (1-2 queries)
2. Vérification des drivers disponibles (2-3 queries)
3. Chargement des bookings (1 query, résultat vide)
4. Vérification des contraintes (2-3 queries)
5. Queries de métadonnées et configuration (5-7 queries)

---

## 🚨 PROBLÈMES IDENTIFIÉS ET RÉSOLUS

### 1. Configuration DB Multi-Environnement ✅

**Problème Initial**:
```
TypeError: 'client_encoding' is an invalid keyword argument for Connection()
```

**Cause**: 
- `client_encoding` (PostgreSQL) était passé à SQLite
- Configuration statique ne détectait pas le type de DB

**Solution Implémentée**:
```python
# backend/config.py
class DevelopmentConfig(Config):
    @staticmethod
    def init_app(app):
        db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
        engine_options = dict(Config.SQLALCHEMY_ENGINE_OPTIONS)
        
        if db_uri.startswith('sqlite'):
            engine_options['connect_args'] = {"check_same_thread": False}
        elif db_uri.startswith('postgresql'):
            engine_options['connect_args'] = {"client_encoding": "utf8"}
        
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = engine_options
```

**Résultat**: ✅ Compatible SQLite (local) et PostgreSQL (Docker)

### 2. Bug Script Profiling ✅

**Problème**: `UnboundLocalError: sorted_queries`

**Solution**: Déclaration de la variable avant utilisation conditionnelle
```python
sorted_queries = sorted(queries_log, key=lambda x: x['time'], reverse=True) if queries_log else []
```

---

## 🎯 PROCHAINES ÉTAPES

### Phase 1: Profiling avec Données Réelles
1. **Créer des données de test réalistes**:
   - 50-100 bookings
   - 10-20 drivers actifs
   - Distribution géographique variée

2. **Ré-exécuter le profiling**:
   ```bash
   docker exec atmr-api-1 python scripts/profiling/profile_dispatch.py
   ```

3. **Analyser les résultats**:
   - Identifier les requêtes N+1
   - Mesurer l'impact des calculs OSRM
   - Évaluer les temps de réponse sous charge

### Phase 2: Optimisations Ciblées (Mardi-Mercredi)
1. **Indexation DB**: Créer index sur colonnes fréquemment utilisées
2. **Eager Loading**: Remplacer lazy loading par `joinedload`/`selectinload`
3. **Query Optimization**: Réduire le nombre de queries via JOIN

### Phase 3: Validation (Jeudi-Vendredi)
1. **Benchmarking**: Comparer avant/après optimisations
2. **Documentation**: Mettre à jour le guide d'optimisation
3. **Tests de Régression**: Garantir aucune régression fonctionnelle

---

## 📝 COMMANDES UTILES

### Exécuter le Profiling

**Dans Docker**:
```bash
docker exec atmr-api-1 python scripts/profiling/profile_dispatch.py
```

**Local (SQLite)**:
```bash
cd backend
python scripts/profiling/profile_dispatch.py
```

### Consulter les Résultats

```bash
docker exec atmr-api-1 cat scripts/profiling/profiling_results.txt
```

### Activer Détection N+1 (optionnel)

Modifier `backend/app.py` pour activer `nplusone`:
```python
from nplusone.ext.flask_sqlalchemy import NPlusOne
nplusone = NPlusOne(app)
```

---

## 📚 RÉFÉRENCES

1. **SQLAlchemy Performance**: https://docs.sqlalchemy.org/en/20/orm/queryguide/performance.html
2. **nplusone Documentation**: https://github.com/jmcarp/nplusone
3. **PostgreSQL Indexing**: https://www.postgresql.org/docs/current/indexes.html
4. **Semaine 2 Guide**: `session/Semaine_2/GUIDE_DETAILLE.md`

---

## ✅ VALIDATION

- [x] Outils de profiling installés
- [x] Script de profiling créé et testé
- [x] Configuration DB multi-environnement corrigée
- [x] Rapport baseline généré
- [ ] Données de test créées (À faire: Phase 1)
- [ ] Profiling avec charge réelle (À faire: Phase 1)

---

**Conclusion**: Le système de profiling est opérationnel et prêt pour les tests avec données réelles. La baseline actuelle montre des performances excellentes, mais ne reflète pas encore la charge réelle du système. Les prochaines étapes consistent à créer des données de test représentatives pour identifier les véritables goulots d'étranglement.

**Statut**: ✅ **BASELINE ÉTABLIE** - Prêt pour phase d'optimisation

