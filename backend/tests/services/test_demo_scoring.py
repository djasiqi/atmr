from services.demo.scoring import compute_demo_score, derive_demo_priority


def test_compute_demo_score_high_priority_profile():
    payload = {
        "timing": "immediate",
        "volume_range": "100_plus",
        "organization_type": "transport_company",
        "integration_required": "yes",
    }
    score = compute_demo_score(payload)
    assert score >= 90
    assert derive_demo_priority(score) == "high"


def test_compute_demo_score_low_priority_profile():
    payload = {
        "timing": "exploration",
        "volume_range": "1_5",
        "organization_type": "curatorship",
        "integration_required": "no",
    }
    score = compute_demo_score(payload)
    assert score <= 20
    assert derive_demo_priority(score) == "standard"
