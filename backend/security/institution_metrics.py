"""Métriques Prometheus pour invitations institution."""

try:
    from prometheus_client import Counter

    institution_invitations_total = Counter(
        "institution_invitations_total",
        "Total invitations et notifications institution",
        ["path", "email_type", "result"],
    )
except ImportError:
    institution_invitations_total = None  # type: ignore[misc, assignment]
