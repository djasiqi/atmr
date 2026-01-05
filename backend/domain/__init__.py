# domain/__init__.py
"""Domain layer - DTOs (Data Transfer Objects) pour découpler les services des modèles SQLAlchemy.

Cette couche permet de :
- Réduire le couplage entre services et infrastructure (SQLAlchemy)
- Faciliter les tests (mocking des repositories)
- Améliorer l'évolutivité (migration vers autre ORM sans modifier les services)

⚠️ Refactoring progressif : Les DTOs sont introduits progressivement.
Les services existants continuent d'utiliser les modèles SQLAlchemy directement
jusqu'à ce qu'ils soient refactorés pour utiliser les repositories.
"""

from domain.assignment_dto import AssignmentDTO
from domain.booking_dto import BookingDTO
from domain.company_dto import CompanyDTO
from domain.dispatch_run_dto import DispatchRunDTO
from domain.driver_dto import DriverDTO

__all__ = [
    "AssignmentDTO",
    "BookingDTO",
    "CompanyDTO",
    "DispatchRunDTO",
    "DriverDTO",
]
