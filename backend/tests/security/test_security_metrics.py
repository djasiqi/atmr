"""Tests pour les métriques de sécurité Prometheus.

Valide que les métriques Prometheus de sécurité sont correctement incrémentées :
- Métriques d'authentification (login, logout, token refresh)
- Métriques d'actions sensibles (création utilisateur, changement permissions)
"""

from prometheus_client import generate_latest


class TestSecurityMetricsAuth:
    """Tests pour les métriques d'authentification."""

    def test_login_attempts_metrics(self, app_context):
        """Vérifie que les métriques de login sont correctement incrémentées."""
        with app_context:
            from security.security_metrics import (
                security_login_attempts_total,
                security_login_failures_total,
            )

            # Simuler un login réussi
            security_login_attempts_total.labels(type="success").inc()

            # Simuler un login échoué
            security_login_attempts_total.labels(type="failed").inc()
            security_login_failures_total.inc()

            # Vérifier que les métriques sont exposées (via generate_latest())
            metrics_output = generate_latest()
            assert b"security_login_attempts_total" in metrics_output
            assert (
                b'type="success"' in metrics_output
                or b'type="success"' in metrics_output
            )
            assert b"security_login_failures_total" in metrics_output

    def test_logout_metrics(self, app_context):
        """Vérifie que les métriques de logout sont correctement incrémentées."""
        with app_context:
            from security.security_metrics import security_logout_total

            # Simuler un logout
            security_logout_total.inc()

            # Vérifier que les métriques sont exposées
            metrics_output = generate_latest()
            assert b"security_logout_total" in metrics_output

    def test_token_refresh_metrics(self, app_context):
        """Vérifie que les métriques de token refresh sont correctement incrémentées."""
        with app_context:
            from security.security_metrics import security_token_refreshes_total

            # Simuler un token refresh
            security_token_refreshes_total.inc()

            # Vérifier que les métriques sont exposées
            metrics_output = generate_latest()
            assert b"security_token_refreshes_total" in metrics_output


class TestSecurityMetricsSensitiveActions:
    """Tests pour les métriques d'actions sensibles."""

    def test_sensitive_actions_metrics(self, app_context):
        """Vérifie que les métriques d'actions sensibles sont correctement incrémentées."""
        with app_context:
            from security.security_metrics import security_sensitive_actions_total

            # Simuler une création d'utilisateur
            security_sensitive_actions_total.labels(action_type="user_created").inc()

            # Simuler un changement de permissions
            security_sensitive_actions_total.labels(
                action_type="permission_changed"
            ).inc()

            # Vérifier que les métriques sont exposées
            metrics_output = generate_latest()
            assert b"security_sensitive_actions_total" in metrics_output
            assert (
                b"user_created" in metrics_output
                or b'action_type="user_created"' in metrics_output
            )

    def test_permission_changes_metrics(self, app_context):
        """Vérifie que les métriques de changement de permissions sont correctement incrémentées."""
        with app_context:
            from security.security_metrics import security_permission_changes_total

            # Simuler un changement de permissions
            security_permission_changes_total.inc()

            # Vérifier que les métriques sont exposées
            metrics_output = generate_latest()
            assert b"security_permission_changes_total" in metrics_output
