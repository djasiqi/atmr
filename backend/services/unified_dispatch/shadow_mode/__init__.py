"""
unified_dispatch.shadow_mode - A/B Testing et Shadow Mode

Ce module contient :
- orchestrator.py : Orchestration shadow mode (ancien shadow_mode_orchestrator.py)
- manager.py : Manager shadow mode (déplacé depuis orchestration/)

Le shadow mode permet de comparer deux versions du dispatch en parallèle
sans impacter la production, pour valider de nouveaux algorithmes.

Créé lors du refactoring B1 - 7 janvier 2025
"""

# Exports publics (à compléter lors de la migration)
__all__ = []
