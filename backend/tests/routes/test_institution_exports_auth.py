"""Tests auth handling institution_exports."""

from __future__ import annotations

import pytest
from jwt.exceptions import ExpiredSignatureError

from routes.institution_exports import _reraise_auth_errors


class TestInstitutionExportsAuth:
    def test_reraise_auth_errors_propagates_expired_signature(self):
        with pytest.raises(ExpiredSignatureError):
            _reraise_auth_errors(ExpiredSignatureError("Signature has expired"))

    def test_reraise_auth_errors_propagates_message_only_expired(self):
        with pytest.raises(Exception, match="Signature has expired"):
            _reraise_auth_errors(Exception("Signature has expired"))
