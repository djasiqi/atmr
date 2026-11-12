"""P0-03 — capacités privilegiées pour l'enrollment MFA."""

from types import SimpleNamespace

from models.enums import InstitutionRole, UserRole
from security.mfa_login import is_restricted_mfa_jwt
from security.privileged_access import requires_mfa_enrollment


def test_platform_admin_requires_enrollment():
    user = SimpleNamespace(
        id=1, role=UserRole.ADMIN, institution_role=None, company=None
    )
    assert requires_mfa_enrollment(user) is True


def test_institution_admin_requires_enrollment():
    user = SimpleNamespace(
        id=2,
        role=UserRole.INSTITUTION,
        institution_role=InstitutionRole.ADMIN.value,
        company=None,
    )
    assert requires_mfa_enrollment(user) is True


def test_institution_requester_not_privileged():
    user = SimpleNamespace(
        id=3,
        role=UserRole.INSTITUTION,
        institution_role=InstitutionRole.REQUESTER.value,
        company=None,
    )
    assert requires_mfa_enrollment(user) is False


def test_institution_billing_not_privileged():
    user = SimpleNamespace(
        id=4,
        role=UserRole.INSTITUTION,
        institution_role=InstitutionRole.BILLING.value,
        company=None,
    )
    assert requires_mfa_enrollment(user) is False


def test_company_owner_requires_enrollment():
    user = SimpleNamespace(
        id=10,
        role=UserRole.COMPANY,
        institution_role=None,
        company=SimpleNamespace(user_id=10),
    )
    assert requires_mfa_enrollment(user) is True


def test_company_operator_not_owner():
    user = SimpleNamespace(
        id=11,
        role=UserRole.COMPANY,
        institution_role=None,
        company=SimpleNamespace(user_id=99),
    )
    assert requires_mfa_enrollment(user) is False


def test_driver_and_client_never_forced():
    driver = SimpleNamespace(
        id=20, role=UserRole.DRIVER, institution_role=None, company=None
    )
    client = SimpleNamespace(
        id=21, role=UserRole.CLIENT, institution_role=None, company=None
    )
    assert requires_mfa_enrollment(driver) is False
    assert requires_mfa_enrollment(client) is False


def test_restricted_mfa_jwt_detects_purposes():
    assert is_restricted_mfa_jwt({"purpose": "2fa_challenge"}) is True
    assert is_restricted_mfa_jwt({"purpose": "mfa_enroll"}) is True
    assert is_restricted_mfa_jwt({"purpose": "access", "role": "ADMIN"}) is False
    assert is_restricted_mfa_jwt({"type": "refresh"}) is False
    assert is_restricted_mfa_jwt(None) is False
