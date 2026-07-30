from __future__ import annotations

from repositories.booking_repository import BookingRepository


def test_find_models_by_driver_and_company_paginated_uses_sql_limit_offset(
    db, sample_company, sample_driver, factory_booking
):
    """Pagination SQL (LIMIT/OFFSET) de l'historique chauffeur (Lot 5 perf) :
    pas de dump complet en mémoire avant découpage de la page.
    """
    from models.enums import BookingStatus

    for _ in range(7):
        factory_booking(
            company=sample_company,
            driver_id=sample_driver.id,
            status=BookingStatus.COMPLETED,
        )
    db.session.flush()

    repo = BookingRepository()

    page1, total = repo.find_models_by_driver_and_company_paginated(
        sample_driver.id,
        sample_company.id,
        statuses=[BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED],
        page=1,
        per_page=3,
    )
    assert total == 7
    assert len(page1) == 3

    page2, total2 = repo.find_models_by_driver_and_company_paginated(
        sample_driver.id,
        sample_company.id,
        statuses=[BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED],
        page=2,
        per_page=3,
    )
    assert total2 == 7
    assert len(page2) == 3

    # Pas de chevauchement entre pages.
    page1_ids = {b.id for b in page1}
    page2_ids = {b.id for b in page2}
    assert page1_ids.isdisjoint(page2_ids)

    page3, total3 = repo.find_models_by_driver_and_company_paginated(
        sample_driver.id,
        sample_company.id,
        statuses=[BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED],
        page=3,
        per_page=3,
    )
    assert total3 == 7
    assert len(page3) == 1
