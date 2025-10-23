# Architecture du Système de Dispatch Unifié

## Vue d'ensemble

Le système de dispatch unifié est conçu pour optimiser l'assignation automatique de chauffeurs aux réservations de transport. Il combine des algorithmes heuristiques rapides avec des solveurs d'optimisation pour fournir des solutions de qualité en temps réel.

## Architecture générale

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   External      │
│   (React)       │    │   (Flask)       │    │   Services      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ UnifiedDispatch │◄──►│ dispatch_routes │◄──►│ OSRM            │
│ Components      │    │ engine.py       │    │ Redis           │
│ Hooks           │    │ heuristics.py   │    │ Celery          │
│ Services        │    │ queue.py        │    │ PostgreSQL      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Composants principaux

### 1. Engine (`engine.py`)

**Responsabilité**: Orchestrateur principal du système de dispatch.

**Fonctions clés**:

- `run()`: Point d'entrée principal pour lancer un dispatch
- `_acquire_day_lock()`: Gestion des verrous distribués Redis
- `_analyze_unassigned_reasons()`: Analyse des raisons de non-assignation

**Flux d'exécution**:

1. Acquisition du verrou distribué
2. Construction du problème (bookings + drivers)
3. Exécution de l'heuristique
4. Exécution du solver (si mode auto)
5. Application des assignations
6. Libération du verrou
7. Retour des métriques

### 2. Heuristics (`heuristics.py`)

**Responsabilité**: Algorithmes d'assignation rapides basés sur des règles.

**Fonctions clés**:

- `assign_heuristic()`: Algorithme glouton principal
- `_can_be_pooled()`: Détection de regroupement de courses
- `_score_driver_booking()`: Calcul du score d'assignation

**Algorithme**:

1. Tri des bookings par priorité (urgences, réguliers)
2. Pour chaque booking, évaluation de tous les chauffeurs disponibles
3. Assignation au chauffeur avec le meilleur score
4. Mise à jour des contraintes (capacité, horaires)

### 3. Queue (`queue.py`)

**Responsabilité**: Gestion de la file d'attente asynchrone des dispatches.

**Fonctions clés**:

- `trigger()`: Déclenchement d'un dispatch
- `_enqueue_celery_task()`: Enfilage des tâches Celery
- Anti-duplication des runs identiques

### 4. Settings (`settings.py`)

**Responsabilité**: Configuration centralisée et paramétrable.

**Classes principales**:

- `Settings`: Configuration globale
- `ServiceTimesSettings`: Temps de service
- `PoolingSettings`: Paramètres de regroupement
- `HeuristicWeights`: Poids des critères d'optimisation

### 5. Data (`data.py`)

**Responsabilité**: Construction du problème d'optimisation.

**Fonctions clés**:

- `build_problem()`: Construction du problème VRPTW
- `_build_driver_windows()`: Fenêtres de travail des chauffeurs
- `_build_booking_constraints()`: Contraintes des réservations

## Flux de données

### 1. Déclenchement d'un dispatch

```
Frontend → POST /company_dispatch/run
         ↓
    dispatch_routes.py
         ↓
    queue.trigger()
         ↓
    Celery Task
         ↓
    engine.run()
```

### 2. Exécution du dispatch

```
engine.run()
    ↓
1. _acquire_day_lock() [Redis]
    ↓
2. _build_problem() [data.py]
    ↓
3. assign_heuristic() [heuristics.py]
    ↓
4. assign_solver() [solver.py] (si mode auto)
    ↓
5. _apply_and_emit() [engine.py]
    ↓
6. _release_day_lock() [Redis]
```

### 3. Mise à jour temps réel

```
WebSocket Events:
- dispatch_run_completed
- booking_updated
- new_booking
```

## Gestion des verrous

### Verrous distribués Redis

**Clé**: `dispatch:lock:{company_id}:{day_str}`
**TTL**: 300 secondes (5 minutes)
**Usage**: Éviter les runs concurrents pour la même entreprise/jour

### Anti-duplication

**Clé**: `dispatch:enqueued:{company_id}:{params_hash}`
**TTL**: 300 secondes
**Usage**: Éviter les runs identiques en cours

## Cache OSRM

### Cache par matrice journalière

**Clé**: `osrm:matrix:{date}:{matrix_hash}`
**TTL**: 7200 secondes (2 heures)
**Usage**: Réduire les appels OSRM répétés

### Cache par paire de points

**Clé**: `osrm:cache:{date}:{origin_hash}:{dest_hash}`
**TTL**: 7200 secondes
**Usage**: Cache des distances/temps entre points

## Métriques et observabilité

### Métriques collectées

- Taux d'assignation
- Temps d'exécution (heuristique + solver)
- Nombre d'appels OSRM
- Latence moyenne OSRM
- Raisons de non-assignation détaillées

### Endpoints de monitoring

- `GET /company_dispatch_health/health`: Santé du système
- `GET /company_dispatch_health/health/trends`: Tendances de performance

## Sécurité

### Validation des paramètres

- Schémas Marshmallow pour validation des entrées
- Validation des dates (format YYYY-MM-DD)
- Validation des modes de dispatch

### Gestion des erreurs

- Circuit breaker pour OSRM
- Fallback vers distances euclidiennes
- Logging structuré avec niveaux appropriés

## Performance

### Optimisations implémentées

1. **Cache Redis**: Réduction des appels OSRM de ~60%
2. **Verrous distribués**: Évite les conflits en environnement multi-workers
3. **Anti-duplication**: Évite les runs redondants
4. **Singleflight**: Regroupe les appels OSRM concurrents

### Exigences de performance

- Dispatch de 50 bookings avec 10 chauffeurs: < 5 secondes
- Taux d'assignation: > 85%
- Disponibilité OSRM: > 99%

## Évolutivité

### Architecture modulaire

- Séparation claire des responsabilités
- Interfaces bien définies entre composants
- Configuration externalisée

### Extensibilité

- Nouveaux algorithmes d'optimisation
- Critères de scoring personnalisables
- Intégration de nouveaux services externes

## Déploiement

### Environnements

- **Développement**: Mode synchrone, logs détaillés
- **Production**: Mode asynchrone, monitoring activé
- **Test**: Mocks des services externes

### Dépendances

- Redis (verrous + cache)
- Celery (tâches asynchrones)
- OSRM (calculs de distance/temps)
- PostgreSQL (persistance des données)
