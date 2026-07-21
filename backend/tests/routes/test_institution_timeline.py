"""Tests auth handling institution_timeline."""

from __future__ import annotations

import pytest
from jwt.exceptions import ExpiredSignatureError

from routes.institution_timeline import _reraise_auth_errors


class TestInstitutionTimelineAuth:
    def test_reraise_auth_errors_propagates_expired_signature(self):
        with pytest.raises(ExpiredSignatureError):
            _reraise_auth_errors(ExpiredSignatureError("Signature has expired"))
