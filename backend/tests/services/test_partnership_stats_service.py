# tests/services/test_partnership_stats_service.py
"""Tests unitaires pour PartnershipStatsService."""

from unittest.mock import MagicMock, patch

import pytest

from services.partnerships.exceptions import StatsComputationError
from services.partnerships.stats import PartnershipStatsService


class TestPartnershipStatsService:
    """Tests pour PartnershipStatsService."""

    def test_get_partnership_stats_raises_stats_error_when_impl_fails(self):
        """get_partnership_stats encapsule toute exception en StatsComputationError."""
        with patch.object(
            PartnershipStatsService,
            "_get_partnership_stats_impl",
            side_effect=AttributeError(
                "type object 'BookingTransfer' has no attribute 'transfer_price'"
            ),
        ):
            partnership = MagicMock()
            partnership.id = 1
            company_id = 42

            with pytest.raises(StatsComputationError) as exc_info:
                PartnershipStatsService.get_partnership_stats(
                    partnership, company_id
                )

        assert "Impossible de calculer les statistiques du partenariat" in str(
            exc_info.value
        )
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, AttributeError)
