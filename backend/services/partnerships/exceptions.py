# services/partnerships/exceptions.py
"""Exceptions métier pour les services partenariats."""


class StatsComputationError(Exception):
    """
    Erreur métier lors du calcul des statistiques de partenariats.

    À utiliser lorsque les données ou le modèle ne permettent pas
    un calcul fiable (champ manquant, relation invalide, etc.).
    """

    pass
