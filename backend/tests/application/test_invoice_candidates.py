"""Agrégation du sélecteur patient, sans base ni cache."""

from types import SimpleNamespace

from application.invoices.invoice_candidates import group_patient_invoice_candidates


def _booking(**overrides):
    payload = {
        "id": 1,
        "client_id": 10,
        "institution_patient_id": None,
        "billing_party_id": 3,
        "amount": 45,
        "status": "COMPLETED",
        "cancellation_fee_amount": None,
        "customer_name": "Jean Dupont",
        "is_return": False,
        "created_via": None,
        "client": None,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_regroupe_deux_courses_du_meme_patient():
    rows = group_patient_invoice_candidates(
        [
            _booking(id=1, amount=45),
            _booking(id=2, amount=30, customer_name="Jean Dupont"),
        ]
    )
    assert len(rows) == 1
    assert rows[0]["billable_count"] == 2
    assert rows[0]["amount"] == 75
    assert rows[0]["client_id"] == 10
    assert rows[0]["billing_party_id"] == 3
    assert rows[0]["id"] == "client:10|billing_party:3"
    assert rows[0]["name"] == "Jean Dupont"


def test_separe_le_patient_institution_du_client_porteur():
    rows = group_patient_invoice_candidates(
        [
            _booking(id=1, client_id=23, institution_patient_id=145, amount=320),
            _booking(
                id=2,
                client_id=23,
                institution_patient_id=None,
                customer_name="Porteur",
                amount=10,
            ),
        ]
    )
    by_name = {row["name"]: row for row in rows}
    assert by_name["Jean Dupont"]["institution_patient_id"] == 145
    assert by_name["Jean Dupont"]["billable_count"] == 1
    assert by_name["Porteur"]["institution_patient_id"] is None


def test_ignore_un_sujet_sans_payeur():
    rows = group_patient_invoice_candidates(
        [_booking(billing_party_id=None, customer_name="Sans payeur")]
    )
    assert rows == []


def test_annulation_utilise_les_frais_pas_le_montant_course():
    rows = group_patient_invoice_candidates(
        [
            _booking(
                status="CANCELED",
                amount=80,
                cancellation_fee_amount=20,
                customer_name="Annulé",
            )
        ]
    )
    assert rows[0]["amount"] == 20


def test_invalidation_ne_touche_que_la_periode_de_l_entreprise(monkeypatch):
    store: dict[str, str] = {}

    class _Redis:
        def delete(self, key):
            store.pop(key, None)

    monkeypatch.setattr("ext.redis_client", _Redis())
    from application.invoices.invoice_candidates import (
        invalidate_patient_invoice_candidates,
        patient_candidates_cache_key,
    )

    keep_company = patient_candidates_cache_key(2, 2026, 9)
    drop = patient_candidates_cache_key(1, 2026, 9)
    keep_month = patient_candidates_cache_key(1, 2026, 8)
    store[keep_company] = "{}"
    store[drop] = "{}"
    store[keep_month] = "{}"
    invalidate_patient_invoice_candidates(1, 2026, 9)
    assert drop not in store
    assert store[keep_company] == "{}"
    assert store[keep_month] == "{}"


def test_seconde_preparation_ignore_la_course_deja_facturee():
    from types import SimpleNamespace

    from application.invoices.billing_opportunities import bookings_still_uninvoiced

    original = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    locked = [
        SimpleNamespace(id=1, invoice_line_id=90),
        SimpleNamespace(id=2, invoice_line_id=None),
    ]
    kept = bookings_still_uninvoiced(original, locked)
    assert [row.id for row in kept] == [2]
