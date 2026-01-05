# backend/services/unified_dispatch/orchestration/__init__.py
"""Module d'orchestration pour le dispatch unifié.

Ce module centralise la logique d'orchestration du dispatch en suivant
le principe de responsabilité unique (SRP). Chaque composant a une
responsabilité claire et bien définie.

Architecture
===========

Le module est organisé en plusieurs composants spécialisés :

1. **DispatchOrchestrator** : Point d'entrée principal
   - Coordonne toutes les étapes du dispatch
   - Gère le verrouillage Redis pour l'idempotence
   - Orchestre les différents managers

2. **DispatchInitializer** : Initialisation et validation
   - Recherche et validation de la Company
   - Configuration des settings avec overrides

3. **DispatchRunManager** : Gestion du cycle de vie DispatchRun
   - Création/réutilisation de DispatchRun
   - Mise à jour du statut
   - Finalisation

4. **ProblemBuilder** : Construction du problème VRPTW
   - Récupération des données (bookings, drivers)
   - Validation des coordonnées géographiques
   - Construction du problème

5. **ClusteringManager** : Clustering géographique
   - Décision d'utiliser le clustering
   - Création de zones géographiques
   - Dispatch par zones

6. **PipelineExecutor** : Exécution du pipeline de dispatch
   - Séparation drivers réguliers/urgence
   - Exécution heuristique → solver → fallback
   - Intégration Shadow Mode

7. **ShadowModeManager** : Gestion du mode shadow
   - Décision d'appliquer les suggestions RL
   - Génération et stockage des suggestions

8. **AssignmentApplierWrapper** : Application des assignations
   - Application en base de données
   - Émission d'événements Socket.IO

9. **MetricsFinalizer** : Finalisation des métriques
   - Calcul des métriques agrégées
   - Analyse des raisons de non-assignation
   - Enregistrement Prometheus (optionnel)
   - Monitoring KPI et backout RL

10. **ResultBuilder** : Construction du résultat final
    - Sérialisation des entités (assignments, bookings, drivers)
    - Construction du DispatchResult
    - Conversion en format dict pour l'API

11. **Utils** : Fonctions utilitaires
    - Conversion de dates
    - Conversion sécurisée d'entiers

Flux d'exécution
===============

Le flux d'exécution suit cette séquence :

1. Initialisation (DispatchInitializer)
2. Verrouillage Redis (idempotence)
3. Gestion DispatchRun (DispatchRunManager)
4. Construction du problème (ProblemBuilder)
5. Clustering optionnel (ClusteringManager)
6. Exécution du pipeline (PipelineExecutor)
   - Shadow Mode intégré (ShadowModeManager)
7. Application des assignations (AssignmentApplierWrapper)
8. Finalisation des métriques (MetricsFinalizer)
   - Construction du résultat (ResultBuilder)

Side-effects
===========

- **Base de données** : Lecture/écriture (Company, DispatchRun, Assignment, Booking)
- **Redis** : Verrous distribués pour éviter runs concurrents
- **Socket.IO** : Émissions d'événements temps réel
- **Métriques** : Prometheus, logging

Exemple d'utilisation
=====================

```python
from services.unified_dispatch.orchestration import DispatchOrchestrator

orchestrator = DispatchOrchestrator()
result = orchestrator.execute(
    company_id=1,
    for_date="2025-01-13",
    mode="auto"
)
```
"""

from services.unified_dispatch.orchestration.dispatch_orchestrator import (
    DispatchOrchestrator,
)

__all__ = ["DispatchOrchestrator"]
