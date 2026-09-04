"""Gate E2E — batch patients institution (idempotence HTTP + concurrence).

Exécution :
    docker compose -f docker-compose.test.yml run --rm backend_tests sh -c \\
      "flask db upgrade heads && python -m pytest \\
       tests/e2e/test_e2e_institution_patient_batch.py -v"
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Barrier

import pytest
from flask_jwt_extended import create_access_token

from ext import db as ext_db
from models import Invoice, User
from models.enums import InvoiceBillingStrategy
from tests.e2e.helpers.institution_invoice_plan_lha import (
    PERIOD_MONTH,
    PERIOD_YEAR,
    build_lha_august_2026_world,
)
from tests.e2e.helpers.institution_patient_batch_world import (
    extend_lha_world_for_patient_batch,
)

pytestmark = pytest.mark.e2e


@pytest.fixture
def batch_world(db):
    return extend_lha_world_for_patient_batch(db, build_lha_august_2026_world(db))


def _headers(world) -> dict[str, str]:
    user = ext_db.session.get(User, world["transport"].user_id)
    assert user is not None
    token = create_access_token(
        identity=str(user.public_id),
        additional_claims={
            "role": user.role.value,
            "company_id": world["transport"].id,
            "aud": "atmr-api",
        },
    )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _url(world) -> str:
    return (
        f"/api/v1/invoices/companies/{world['transport'].id}"
        "/invoices/institution-patient-batch"
    )


def _payload(world, patient_ids: list[int]) -> dict:
    return {
        "year": PERIOD_YEAR,
        "month": PERIOD_MONTH,
        "clinic_company_id": world["clinic"].id,
        "clinic_client_id": world["clinic_client"].id,
        "institution_patient_ids": patient_ids,
    }


def _s1_count(world) -> int:
    return Invoice.query.filter(
        Invoice.company_id == world["transport"].id,
        Invoice.period_year == PERIOD_YEAR,
        Invoice.period_month == PERIOD_MONTH,
        Invoice.billing_strategy == InvoiceBillingStrategy.S1_PATIENT,
    ).count()


class TestE2EInstitutionPatientBatch:
    def test_http_first_then_identical_reuse(self, client, db, batch_world):
        ids = [
            batch_world["patients"]["cavadini"].id,
            batch_world["patients"]["moretti"].id,
        ]
        url = _url(batch_world)
        headers = _headers(batch_world)
        payload = _payload(batch_world, ids)

        first = client.post(url, json=payload, headers=headers)
        assert first.status_code == 200, first.get_json()
        body = (first.get_json() or {}).get("data") or {}
        assert body["created_count"] == 2
        assert body["reused_count"] == 0
        assert body["failed_count"] == 0
        first_invoice_ids = {row["invoice_id"] for row in body["invoices"]}
        assert len(first_invoice_ids) == 2
        after_first = _s1_count(batch_world)

        second = client.post(url, json=payload, headers=headers)
        assert second.status_code == 200, second.get_json()
        body2 = (second.get_json() or {}).get("data") or {}
        assert body2["created_count"] == 0
        assert body2["reused_count"] == 2
        assert {row["invoice_id"] for row in body2["invoices"]} == first_invoice_ids
        assert _s1_count(batch_world) == after_first

    def test_http_concurrent_no_duplicate(self, app, db, batch_world):
        ids = [
            batch_world["patients"]["cavadini"].id,
            batch_world["patients"]["moretti"].id,
        ]
        url = _url(batch_world)
        headers = _headers(batch_world)
        payload = _payload(batch_world, ids)
        barrier = Barrier(2)

        def _post_once() -> tuple[int, dict]:
            barrier.wait(timeout=30)
            with app.test_client() as thread_client:
                response = thread_client.post(url, json=payload, headers=headers)
                return response.status_code, response.get_json() or {}

        results: list[tuple[int, dict]] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_post_once) for _ in range(2)]
            for future in as_completed(futures):
                results.append(future.result())

        assert all(status == 200 for status, _ in results), results
        bodies = [(body.get("data") or {}) for _, body in results]
        created_reused = [
            (int(b.get("created_count") or 0), int(b.get("reused_count") or 0))
            for b in bodies
        ]
        assert all(c + r == 2 for c, r in created_reused), created_reused
        assert sum(c for c, _ in created_reused) == 2
        invoice_ids = {
            row["invoice_id"] for body in bodies for row in (body.get("invoices") or [])
        }
        assert len(invoice_ids) == 2
        assert _s1_count(batch_world) == 2
