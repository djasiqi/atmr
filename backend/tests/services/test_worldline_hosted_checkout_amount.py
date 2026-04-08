from services.worldline.money import chf_amount_to_cents


def test_chf_amount_to_cents():
    assert chf_amount_to_cents(50.0) == 5000
    assert chf_amount_to_cents(0.5) == 50
    assert chf_amount_to_cents(12.34) == 1234
