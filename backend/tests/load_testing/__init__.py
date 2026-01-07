"""
Tests de charge (Load Testing) - Locust

Ce module contient les tests de charge pour valider la performance et la résilience
du système de dispatch sous différents scénarios.

Scénarios disponibles :
- Scénario 1 : Charge standard (100 bookings x 50 drivers)
- Scénario 2 : Multi-entreprises (10 entreprises parallèles)
- Scénario 3 : OSRM lent (résilience 500ms latency)

## Installation

```bash
cd backend
pip install locust
```

## Usage

Voir README.md dans ce dossier pour les instructions détaillées.
"""

# Note: __all__ est vide car ce sont des scripts Locust,
# pas des modules Python importables
__all__: list[str] = []
