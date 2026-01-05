"""Repository pour l'accès aux données FavoritePlace."""

from models.medical import FavoritePlace

logger = __import__("logging").getLogger(__name__)


class FavoritePlaceRepository:
    """Repository pour l'accès aux données FavoritePlace."""

    def find_by_company_id_with_label_search(
        self, company_id: int, search_query: str, limit: int = 6
    ) -> list[FavoritePlace]:
        """Trouve les lieux favoris d'une entreprise avec recherche sur le label.

        Args:
            company_id: ID de l'entreprise
            search_query: Requête de recherche (pattern LIKE)
            limit: Nombre maximum de résultats (défaut: 6)

        Returns:
            Liste de FavoritePlace triées par label asc
        """
        like_q = f"%{search_query.lower()}%"
        return (
            FavoritePlace.query.filter(
                FavoritePlace.company_id == company_id,
                FavoritePlace.label.ilike(like_q),
            )
            .order_by(FavoritePlace.label.asc())
            .limit(limit)
            .all()
        )
