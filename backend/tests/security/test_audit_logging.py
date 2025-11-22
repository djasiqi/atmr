"""Tests de sécurité pour l'audit logging.

Valide que l'audit logging fonctionne correctement pour :
- Login/logout/token refresh
- Création/modification d'utilisateurs
- Changements de permissions
"""

from flask import current_app

from models import User, UserRole
from security.audit_log import AuditLogger


class TestAuditLoggingAuth:
    """Tests pour l'audit logging des actions d'authentification."""

    def test_login_success_logged(self, app_context, db, sample_user):
        """Vérifie que le login réussi est loggé dans AuditLog."""
        with app_context:
            # Simuler un login réussi
            from flask import request

            with current_app.test_request_context(
                path="/auth/login",
                method="POST",
                headers={"User-Agent": "test-agent"},
                environ_base={"REMOTE_ADDR": "127.0.0.1"},
            ):
                # Logger l'action de login
                audit_log = AuditLogger.log_action(
                    action_type="login_success",
                    action_category="security",
                    user_id=sample_user.id,
                    user_type=sample_user.role.value if sample_user.role else "unknown",
                    result_status="success",
                    action_details={
                        "email": "test@example.com",
                        "role": sample_user.role.value if sample_user.role else None,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )

                db.session.commit()

                # Vérifier que l'audit log existe
                assert audit_log is not None
                assert audit_log.id is not None
                assert audit_log.action_type == "login_success"
                assert audit_log.action_category == "security"
                assert audit_log.user_id == sample_user.id
                assert audit_log.result_status == "success"
                assert audit_log.ip_address == "127.0.0.1"
                assert audit_log.user_agent == "test-agent"

    def test_login_failed_logged(self, app_context, db):
        """Vérifie que le login échoué est loggé dans AuditLog."""
        with app_context:
            from flask import request

            with current_app.test_request_context(
                path="/auth/login",
                method="POST",
                headers={"User-Agent": "test-agent"},
                environ_base={"REMOTE_ADDR": "127.0.0.1"},
            ):
                # Logger l'action de login échoué
                audit_log = AuditLogger.log_action(
                    action_type="login_failed",
                    action_category="security",
                    user_type="unknown",
                    result_status="failure",
                    result_message="Email ou mot de passe invalide",
                    action_details={"email": "test@example.com", "reason": "invalid_credentials"},
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )

                db.session.commit()

                # Vérifier que l'audit log existe
                assert audit_log is not None
                assert audit_log.id is not None
                assert audit_log.action_type == "login_failed"
                assert audit_log.action_category == "security"
                assert audit_log.result_status == "failure"
                assert audit_log.user_id is None  # Pas d'utilisateur pour un échec
                assert audit_log.ip_address == "127.0.0.1"

    def test_logout_logged(self, app_context, db, sample_user):
        """Vérifie que le logout est loggé dans AuditLog."""
        with app_context:
            from flask import request

            with current_app.test_request_context(
                path="/auth/logout",
                method="POST",
                headers={"User-Agent": "test-agent"},
                environ_base={"REMOTE_ADDR": "127.0.0.1"},
            ):
                # Logger l'action de logout
                audit_log = AuditLogger.log_action(
                    action_type="logout",
                    action_category="security",
                    user_id=sample_user.id,
                    user_type=sample_user.role.value if sample_user.role else "unknown",
                    result_status="success",
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )

                db.session.commit()

                # Vérifier que l'audit log existe
                assert audit_log is not None
                assert audit_log.action_type == "logout"
                assert audit_log.user_id == sample_user.id
                assert audit_log.result_status == "success"


class TestAuditLoggingSensitiveActions:
    """Tests pour l'audit logging des actions sensibles."""

    def test_user_created_logged(self, app_context, db, sample_user, sample_company):
        """Vérifie que la création d'utilisateur est loggée."""
        with app_context:
            from flask import request

            with current_app.test_request_context(
                path="/companies/me/clients",
                method="POST",
                headers={"User-Agent": "test-agent"},
                environ_base={"REMOTE_ADDR": "127.0.0.1"},
            ):
                # Simuler la création d'un nouveau client
                new_user = User()
                new_user.username = "newclient"
                new_user.email = "newclient@example.com"
                new_user.role = UserRole.client
                db.session.add(new_user)
                db.session.flush()

                # Logger l'action de création
                audit_log = AuditLogger.log_action(
                    action_type="user_created",
                    action_category="security",
                    user_id=sample_user.id,
                    user_type=sample_user.role.value if sample_user.role else "unknown",
                    result_status="success",
                    action_details={
                        "created_user_id": new_user.id,
                        "created_user_email": "newclient@example.com",
                        "created_user_role": "client",
                    },
                    company_id=sample_company.id if sample_company else None,
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )

                db.session.commit()

                # Vérifier que l'audit log existe
                assert audit_log is not None
                assert audit_log.action_type == "user_created"
                assert audit_log.company_id == sample_company.id if sample_company else None

    def test_permission_changed_logged(self, app_context, db, sample_user):
        """Vérifie que le changement de permission est loggé."""
        with app_context:
            from flask import request

            with current_app.test_request_context(
                path="/admin/users/1/role",
                method="PUT",
                headers={"User-Agent": "test-agent"},
                environ_base={"REMOTE_ADDR": "127.0.0.1"},
            ):
                # Logger le changement de rôle
                audit_log = AuditLogger.log_action(
                    action_type="permission_changed",
                    action_category="security",
                    user_id=sample_user.id,
                    user_type="admin",
                    result_status="success",
                    action_details={
                        "modified_user_id": 1,
                        "old_role": "CLIENT",
                        "new_role": "DRIVER",
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )

                db.session.commit()

                # Vérifier que l'audit log existe
                assert audit_log is not None
                assert audit_log.action_type == "permission_changed"
                assert "old_role" in audit_log.action_details
                assert "new_role" in audit_log.action_details
